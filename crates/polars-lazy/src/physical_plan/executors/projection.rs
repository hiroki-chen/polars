use std::any::Any;

use picachv::plan_argument::Argument;
use picachv::{PlanArgument, ProjectionArgument};
use polars_core::utils::accumulate_dataframes_vertical_unchecked;

use super::*;

/// Take an input Executor (creates the input DataFrame)
/// and a multiple PhysicalExpressions (create the output Series)
pub struct ProjectionExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) cse_exprs: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) expr: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) has_windows: bool,
    pub(crate) input_schema: SchemaRef,
    #[cfg(test)]
    pub(crate) schema: SchemaRef,
    pub(crate) options: ProjectionOptions,
    // Can run all operations elementwise
    pub(crate) streamable: bool,
}

impl ProjectionExec {
    pub(crate) fn new(
        input: Box<dyn Executor>,
        cse_exprs: Vec<Arc<dyn PhysicalExpr>>,
        expr: Vec<Arc<dyn PhysicalExpr>>,
        has_windows: bool,
        input_schema: SchemaRef,
        #[cfg(test)] schema: SchemaRef,
        options: ProjectionOptions,
        streamable: bool,
    ) -> Self {
        ProjectionExec {
            input,
            cse_exprs,
            expr,
            has_windows,
            input_schema,
            #[cfg(test)]
            schema,
            options,
            streamable,
        }
    }

    fn execute_impl(
        &mut self,
        state: &ExecutionState,
        mut df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        // Vertical and horizontal parallelism.
        let df =
            if false && self.streamable && df.n_chunks() > 1 && df.height() > 0 && self.options.run_parallel
            {
                let chunks = df.split_chunks().collect::<Vec<_>>();
                let iter = chunks.into_par_iter().map(|mut df| {
                    let selected_cols = evaluate_physical_expressions(
                        &mut df,
                        &self.cse_exprs,
                        &self.expr,
                        state,
                        self.has_windows,
                        self.options.run_parallel,
                    )?;
                    check_expand_literals(
                        selected_cols,
                        df.height() == 0,
                        self.options.duplicate_check,
                    )
                });

                let df = POOL.install(|| iter.collect::<PolarsResult<Vec<_>>>())?;
                accumulate_dataframes_vertical_unchecked(df)
            }
            // Only horizontal parallelism.
            else {
                #[allow(clippy::let_and_return)]
                let selected_cols = evaluate_physical_expressions(
                    &mut df,
                    &self.cse_exprs,
                    &self.expr,
                    state,
                    self.has_windows,
                    self.options.run_parallel,
                )?;
                check_expand_literals(
                    selected_cols,
                    df.height() == 0,
                    self.options.duplicate_check,
                )?
            };

        // this only runs during testing and check if the runtime type matches the predicted schema
        #[cfg(test)]
        #[allow(unused_must_use)]
        {
            // TODO: also check the types.
            for (l, r) in df.iter().zip(self.schema.iter_names()) {
                assert_eq!(l.name(), r);
            }
        }

        Ok(df)
    }
}

impl Executor for ProjectionExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                if self.cse_exprs.is_empty() {
                    eprintln!("run ProjectionExec");
                } else {
                    eprintln!("run ProjectionExec with {} CSE", self.cse_exprs.len())
                };
            }
        }

        let df = self.input.execute(state)?;

        let profile_name = if state.has_node_timer() {
            let by = self
                .expr
                .iter()
                .map(|s| {
                    profile_name(
                        s.as_ref(),
                        self.input_schema.as_ref(),
                        !self.cse_exprs.is_empty(),
                    )
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            let name = comma_delimited("select".to_string(), &by);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        let df = if state.has_node_timer() {
            let new_state = state.clone();
            new_state.record(|| self.execute_impl(state, df), profile_name)
        } else {
            self.execute_impl(state, df)
        }?;

        if state.policy_check {
            // Must be executed `after` the input is executed otherwise we have no idea
            // what the uuid of the input is.
            // self.execute_prologue(state)?;
            let plan_arg = PlanArgument {
                argument: Some(Argument::Projection(ProjectionArgument {
                    expressions: self
                        .expr
                        .iter()
                        .map(|e| e.get_uuid().to_bytes_le().to_vec())
                        .collect(),
                })),
                transform_info: state.transform.clone(),
            };
            self.execute_epilogue(state, Some(plan_arg))?;
        }

        Ok(df)
    }
}

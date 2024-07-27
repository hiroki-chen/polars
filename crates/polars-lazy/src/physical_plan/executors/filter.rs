use picachv::plan_argument::Argument;
use picachv::{PlanArgument, SelectArgument, TransformInfo};
use polars_core::utils::accumulate_dataframes_vertical_unchecked;

use super::*;

pub struct FilterExec {
    pub(crate) predicate: Arc<dyn PhysicalExpr>,
    pub(crate) input: Box<dyn Executor>,
    // if the predicate contains a window function
    has_window: bool,
    streamable: bool,
}

fn series_to_mask(s: &Series) -> PolarsResult<&BooleanChunked> {
    s.bool().map_err(|_| {
        polars_err!(
            ComputeError: "filter predicate must be of type `Boolean`, got `{}`", s.dtype()
        )
    })
}

impl FilterExec {
    pub fn new(
        predicate: Arc<dyn PhysicalExpr>,
        input: Box<dyn Executor>,
        has_window: bool,
        streamable: bool,
    ) -> Self {
        Self {
            predicate,
            input,
            has_window,
            streamable,
        }
    }

    fn execute_hor(
        &mut self,
        df: DataFrame,
        state: &mut ExecutionState,
    ) -> PolarsResult<DataFrame> {
        if self.has_window {
            state.insert_has_window_function_flag()
        }

        let s = self.predicate.evaluate(&df, state)?;
        if self.has_window {
            state.clear_window_expr_cache()
        }
        let pred = series_to_mask(&s)?;
        let pred_bool = pred
            .iter()
            .map(|e| e.ok_or(PolarsError::ComputeError("Filter downcast failed".into())))
            .collect::<PolarsResult<Vec<_>>>()?;

        let df = df.filter(pred)?;

        state.transform.replace(
            TransformInfo::from_filter(&pred_bool)
                .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?,
        );

        Ok(df)
    }

    fn execute_chunks(
        &mut self,
        chunks: Vec<DataFrame>,
        state: &mut ExecutionState,
    ) -> PolarsResult<DataFrame> {
        let iter = chunks.into_par_iter().map(|df| {
            let s = self.predicate.evaluate(&df, state)?;

            let pred = series_to_mask(&s)?;
            let pred_bool = pred
                .iter()
                .map(|e| e.ok_or(PolarsError::ComputeError("Filter downcast failed".into())))
                .collect::<PolarsResult<Vec<_>>>()?;
            let df = df.filter(pred)?;

            Ok((df, pred_bool))
        });

        let res = POOL.install(|| iter.collect::<PolarsResult<Vec<_>>>())?;
        let df = res.iter().map(|(df, _)| df.clone()).collect::<Vec<_>>();
        let pred_bool = res
            .iter()
            .map(|(_, pred)| pred.clone())
            .flatten()
            .collect::<Vec<_>>();
        state.transform.replace(
            TransformInfo::from_filter(&pred_bool)
                .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?,
        );
        Ok(accumulate_dataframes_vertical_unchecked(df))
    }

    fn execute_impl(
        &mut self,
        df: DataFrame,
        state: &mut ExecutionState,
    ) -> PolarsResult<DataFrame> {
        let n_partitions = POOL.current_num_threads();
        // Vertical parallelism.
        if self.streamable && df.height() > 0 {
            if df.n_chunks() > 1 {
                let chunks = df.split_chunks().collect::<Vec<_>>();

                self.execute_chunks(chunks, state)
            } else if df.width() < n_partitions {
                self.execute_hor(df, state)
            } else {
                let chunks = df.split_chunks_by_n(n_partitions, true);
                self.execute_chunks(chunks, state)
            }
        } else {
            self.execute_hor(df, state)
        }
    }
}

impl Executor for FilterExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run FilterExec")
            }
        }
        let df = self.input.execute(state)?;

        let profile_name = if state.has_node_timer() {
            Cow::Owned(format!(".filter({})", &self.predicate.as_ref()))
        } else {
            Cow::Borrowed("")
        };

        let df = state
            .clone()
            .record(
                || {
                    let df = self.execute_impl(df, state);
                    if state.verbose() {
                        eprintln!("dataframe filtered");
                    }
                    df
                },
                profile_name,
            )
            .unwrap();
        if state.policy_check {
            let plan_arg = PlanArgument {
                argument: Some(Argument::Select(SelectArgument {
                    pred_uuid: self.predicate.get_uuid().to_bytes_le().to_vec(),
                })),
                transform_info: state.transform.clone(),
            };
            self.execute_epilogue(state, Some(plan_arg))?;
        }

        Ok(df)
    }
}

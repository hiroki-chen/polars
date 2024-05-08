use picachv::group_by_proxy::Groups;
use picachv::plan_argument::Argument;
use picachv::{AggregateArgument, GroupByProxy, PlanArgument};
use rayon::prelude::*;

use super::*;

pub(super) fn evaluate_aggs(
    df: &DataFrame,
    aggs: &[Arc<dyn PhysicalExpr>],
    groups: &GroupsProxy,
    state: &ExecutionState,
) -> PolarsResult<Vec<Series>> {
    // POOL.install(|| {
    //     aggs.par_iter()
    //         .map(|expr| {
    //             let agg = expr.evaluate_on_groups(df, groups, state)?.finalize();
    //             polars_ensure!(agg.len() == groups.len(), agg_len = agg.len(), groups.len());
    //             Ok(agg)
    //         })
    //         .collect::<PolarsResult<Vec<_>>>()
    // })

    // For debugging.
    aggs.iter()
        .map(|expr| {
            let agg = expr.evaluate_on_groups(df, groups, state)?.finalize();
            polars_ensure!(agg.len() == groups.len(), agg_len = agg.len(), groups.len());
            Ok(agg)
        })
        .collect::<PolarsResult<Vec<_>>>()
}

/// Take an input Executor and a multiple expressions
pub struct GroupByExec {
    input: Box<dyn Executor>,
    keys: Vec<Arc<dyn PhysicalExpr>>,
    aggs: Vec<Arc<dyn PhysicalExpr>>,
    apply: Option<Arc<dyn DataFrameUdf>>,
    maintain_order: bool,
    input_schema: SchemaRef,
    slice: Option<(i64, usize)>,
}

impl GroupByExec {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input: Box<dyn Executor>,
        keys: Vec<Arc<dyn PhysicalExpr>>,
        aggs: Vec<Arc<dyn PhysicalExpr>>,
        apply: Option<Arc<dyn DataFrameUdf>>,
        maintain_order: bool,
        input_schema: SchemaRef,
        slice: Option<(i64, usize)>,
    ) -> Self {
        Self {
            input,
            keys,
            aggs,
            apply,
            maintain_order,
            input_schema,
            slice,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn group_by_helper(
    mut df: DataFrame,
    keys: Vec<Series>,
    aggs: &[Arc<dyn PhysicalExpr>],
    apply: Option<Arc<dyn DataFrameUdf>>,
    state: &mut ExecutionState,
    maintain_order: bool,
    slice: Option<(i64, usize)>,
) -> PolarsResult<DataFrame> {
    println!("group_by_helper");

    df.as_single_chunk_par();
    let gb = df.group_by_with_series(keys, true, maintain_order)?;

    // todo: add a check here? We may need it.
    if let Some(f) = apply {
        return gb.apply(move |df| f.call_udf(df));
    }

    let mut groups = gb.get_groups();

    // Do not clone here since doing so will cause data corruption (we don't know why).
    if let GroupsProxy::Idx(idx) = groups {
        state.last_used_groupby.0 = idx.first().iter().map(|&e| e as u64).collect();
        state.last_used_groupby.1 = idx
            .all()
            .iter()
            .map(|e| e.iter().map(|&e| e as u64).collect())
            .collect();
    }

    #[allow(unused_assignments)]
    // it is unused because we only use it to keep the lifetime of sliced_group valid
    let mut sliced_groups = None;

    if let Some((offset, len)) = slice {
        sliced_groups = Some(groups.slice(offset, len));
        groups = sliced_groups.as_deref().unwrap();
    }

    // let (mut columns, agg_columns) = POOL.install(|| {
    //     let get_columns = || gb.keys_sliced(slice);

    //     let get_agg = || evaluate_aggs(&df, aggs, groups, state);

    //     rayon::join(get_columns, get_agg)
    // });
    let (mut columns, agg_columns) = (
        gb.keys_sliced(slice),
        evaluate_aggs(&df, aggs, groups, state),
    );
    let agg_columns = agg_columns?;

    columns.extend_from_slice(&agg_columns);
    DataFrame::new(columns)
}

impl GroupByExec {
    fn execute_impl(
        &mut self,
        state: &mut ExecutionState,
        df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        println!("execute_impl in GroupByExec");

        let keys = self
            .keys
            .iter()
            .map(|e| e.evaluate(&df, state))
            .collect::<PolarsResult<_>>()?;
        group_by_helper(
            df,
            keys,
            &self.aggs,
            self.apply.take(),
            state,
            self.maintain_order,
            self.slice,
        )
    }
}

impl Executor for GroupByExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run GroupbyExec")
            }
        }
        if state.verbose() {
            eprintln!("keys/aggregates are not partitionable: running default HASH AGGREGATION")
        }
        let df = self.input.execute(state)?;

        let profile_name = if state.has_node_timer() {
            let by = self
                .keys
                .iter()
                .map(|s| Ok(s.to_field(&self.input_schema)?.name))
                .collect::<PolarsResult<Vec<_>>>()?;
            let name = comma_delimited("group_by".to_string(), &by);
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

        let gb = Some(state.last_used_groupby.clone());
        let plan_arg = PlanArgument {
            argument: Some(Argument::Aggregate(AggregateArgument {
                keys: self
                    .keys
                    .iter()
                    .map(|e| e.get_uuid().to_bytes_le().to_vec())
                    .collect(),
                aggs_uuid: self
                    .aggs
                    .iter()
                    .map(|e| e.get_uuid().to_bytes_le().to_vec())
                    .collect(),
                maintain_order: self.maintain_order,
                group_by_proxy: gb.map(|gb| GroupByProxy {
                    first: gb.0,
                    groups: gb
                        .1
                        .into_iter()
                        .map(|e| Groups {
                            group: e.iter().map(|&e| e as u64).collect(),
                        })
                        .collect(),
                }),
                output_schema: self
                    .input_schema
                    .get_names()
                    .into_iter()
                    .map(|s| s.to_string())
                    .collect(),
            })),
        };

        println!("sending plan_arg for groupby: {plan_arg:?}");

        self.execute_epilogue(state, Some(plan_arg))?;

        Ok(df)
    }
}

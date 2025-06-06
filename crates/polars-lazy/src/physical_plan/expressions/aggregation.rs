use std::borrow::Cow;

use arrow::array::*;
use arrow::compute::concatenate::concatenate;
use arrow::legacy::utils::CustomIterTools;
use arrow::offset::Offsets;
use picachv::expr_argument::Argument;
use picachv::native::{build_expr, reify_expression};
use picachv::{AggExpr, ExprArgument};
use polars_core::prelude::*;
use polars_core::utils::NoNull;
#[cfg(feature = "dtype-struct")]
use polars_core::POOL;
#[cfg(feature = "propagate_nans")]
use polars_ops::prelude::nan_propagating_aggregate;
use uuid::Uuid;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::AggState::{AggregatedList, AggregatedScalar};
use crate::prelude::*;

pub(crate) struct AggregationExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) agg_type: GroupByMethod,
    field: Option<Field>,
    expr_uuid: Uuid,
}

impl AggregationExpr {
    pub fn new(
        expr: Arc<dyn PhysicalExpr>,
        agg_type: GroupByMethod,
        field: Option<Field>,
        ctx_id: Uuid,
        policy_check: bool,
    ) -> PolarsResult<Self> {
        let expr_uuid = if policy_check {
            // Wrap into another function.
            let arg = arg_to_expr(&expr, agg_type)?;

            build_expr(ctx_id, arg).map_err(|e| PolarsError::ComputeError(e.to_string().into()))?
        } else {
            Default::default()
        };

        Ok(Self {
            input: expr,
            agg_type,
            field,
            expr_uuid,
        })
    }
}

fn arg_to_expr(
    expr: &Arc<dyn PhysicalExpr>,
    agg_type: GroupByMethod,
) -> PolarsResult<ExprArgument> {
    let input_uuid = expr.get_uuid().to_bytes_le().to_vec();
    let arg = ExprArgument {
        argument: Some(Argument::Agg(AggExpr {
            input_uuid,
            method: match agg_type {
                GroupByMethod::Sum => picachv::GroupByMethod::Sum,
                GroupByMethod::Mean => picachv::GroupByMethod::Mean,
                GroupByMethod::Min => picachv::GroupByMethod::Min,
                GroupByMethod::Max => picachv::GroupByMethod::Max,
                GroupByMethod::Count { .. } => picachv::GroupByMethod::Len,
                GroupByMethod::NUnique => picachv::GroupByMethod::Min,
                _ => polars_bail!(ComputeError: "Aggregation method not supported: {:?}", agg_type),
            } as _,
        })),
    };

    Ok(arg)
}

impl PhysicalExpr for AggregationExpr {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    fn get_name(&self) -> &str {
        "Aggregation"
    }

    fn get_uuid(&self) -> Uuid {
        self.expr_uuid
    }

    fn evaluate(&self, _df: &DataFrame, _state: &ExecutionState) -> PolarsResult<Series> {
        unimplemented!()
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;
        // don't change names by aggregations as is done in polars-core
        let keep_name = ac.series().name().to_string();
        polars_ensure!(!matches!(ac.agg_state(), AggState::Literal(_)), ComputeError: "cannot aggregate a literal");

        if let AggregatedScalar(_) = ac.agg_state() {
            match self.agg_type {
                GroupByMethod::Implode => {},
                _ => {
                    polars_bail!(ComputeError: "cannot aggregate as {}, the column is already aggregated", self.agg_type);
                },
            }
        }

        // SAFETY:
        // groups must always be in bounds.
        let out = unsafe {
            match self.agg_type {
                GroupByMethod::Min => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_min(&groups);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Max => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_max(&groups);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Median => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_median(&groups);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Mean => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_mean(&groups);

                    if state.policy_check {
                        let bytes = inputs_as_arrow(&[&agg_s])?;
                        reify_expression(state.ctx_id, self.expr_uuid, &bytes)
                            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
                    }

                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Sum => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_sum(&groups);

                    if state.policy_check {
                        let bytes = inputs_as_arrow(&[&agg_s])?;
                        reify_expression(state.ctx_id, self.expr_uuid, &bytes)
                            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
                    }

                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Count { include_nulls } => {
                    if include_nulls || ac.series().null_count() == 0 {
                        // a few fast paths that prevent materializing new groups
                        match ac.update_groups {
                            UpdateGroups::WithSeriesLen => {
                                let list = ac
                                    .series()
                                    .list()
                                    .expect("impl error, should be a list at this point");

                                let mut s = match list.chunks().len() {
                                    1 => {
                                        let arr = list.downcast_iter().next().unwrap();
                                        let offsets = arr.offsets().as_slice();

                                        let mut previous = 0i64;
                                        let counts: NoNull<IdxCa> = offsets[1..]
                                            .iter()
                                            .map(|&o| {
                                                let len = (o - previous) as IdxSize;
                                                previous = o;
                                                len
                                            })
                                            .collect_trusted();
                                        counts.into_inner()
                                    },
                                    _ => {
                                        let counts: NoNull<IdxCa> = list
                                            .amortized_iter()
                                            .map(|s| {
                                                if let Some(s) = s {
                                                    s.as_ref().len() as IdxSize
                                                } else {
                                                    1
                                                }
                                            })
                                            .collect_trusted();
                                        counts.into_inner()
                                    },
                                };
                                s.rename(&keep_name);
                                AggregatedScalar(s.into_series())
                            },
                            UpdateGroups::WithGroupsLen => {
                                // no need to update the groups
                                // we can just get the attribute, because we only need the length,
                                // not the correct order
                                let mut ca = ac.groups.group_count();
                                ca.rename(&keep_name);
                                AggregatedScalar(ca.into_series())
                            },
                            // materialize groups
                            _ => {
                                let mut ca = ac.groups().group_count();
                                ca.rename(&keep_name);
                                AggregatedScalar(ca.into_series())
                            },
                        }
                    } else {
                        // TODO: optimize this/and write somewhere else.
                        match ac.agg_state() {
                            AggState::Literal(s) | AggState::AggregatedScalar(s) => {
                                AggregatedScalar(Series::new(
                                    &keep_name,
                                    [(s.len() as IdxSize - s.null_count() as IdxSize)],
                                ))
                            },
                            AggState::AggregatedList(s) => {
                                let ca = s.list()?;
                                let out: IdxCa = ca
                                    .into_iter()
                                    .map(|opt_s| {
                                        opt_s
                                            .map(|s| s.len() as IdxSize - s.null_count() as IdxSize)
                                    })
                                    .collect();
                                AggregatedScalar(rename_series(out.into_series(), &keep_name))
                            },
                            AggState::NotAggregated(s) => {
                                let s = s.clone();
                                let groups = ac.groups();
                                let out: IdxCa = if matches!(s.dtype(), &DataType::Null) {
                                    IdxCa::full(s.name(), 0, groups.len())
                                } else {
                                    match groups.as_ref() {
                                        GroupsProxy::Idx(idx) => {
                                            let s = s.rechunk();
                                            let array = &s.chunks()[0];
                                            let validity = array.validity().unwrap();
                                            idx.iter()
                                                .map(|(_, g)| {
                                                    let mut count = 0 as IdxSize;
                                                    // Count valid values
                                                    g.iter().for_each(|i| {
                                                        count += validity
                                                            .get_bit_unchecked(*i as usize)
                                                            as IdxSize;
                                                    });
                                                    count
                                                })
                                                .collect_ca_trusted_with_dtype(
                                                    &keep_name, IDX_DTYPE,
                                                )
                                        },
                                        GroupsProxy::Slice { groups, .. } => {
                                            // Slice and use computed null count
                                            groups
                                                .iter()
                                                .map(|g| {
                                                    let start = g[0];
                                                    let len = g[1];
                                                    len - s
                                                        .slice(start as i64, len as usize)
                                                        .null_count()
                                                        as IdxSize
                                                })
                                                .collect_ca_trusted_with_dtype(
                                                    &keep_name, IDX_DTYPE,
                                                )
                                        },
                                    }
                                };
                                AggregatedScalar(out.into_series())
                            },
                        }
                    }
                },
                GroupByMethod::First => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_first(&groups);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Last => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_last(&groups);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::NUnique => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_n_unique(&groups);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Implode => {
                    // if the aggregation is already
                    // in an aggregate flat state for instance by
                    // a mean aggregation, we simply convert to list
                    //
                    // if it is not, we traverse the groups and create
                    // a list per group.
                    let s = match ac.agg_state() {
                        // mean agg:
                        // -> f64 -> list<f64>
                        AggState::AggregatedScalar(s) => s.reshape(&[-1, 1]).unwrap(),
                        _ => {
                            let agg = ac.aggregated();
                            agg.as_list().into_series()
                        },
                    };
                    AggregatedList(rename_series(s, &keep_name))
                },
                GroupByMethod::Groups => {
                    let mut column: ListChunked = ac.groups().as_list_chunked();
                    column.rename(&keep_name);
                    AggregatedScalar(column.into_series())
                },
                GroupByMethod::Std(ddof) => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_std(&groups, ddof);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Var(ddof) => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_var(&groups, ddof);
                    AggregatedScalar(rename_series(agg_s, &keep_name))
                },
                GroupByMethod::Quantile(_, _) => {
                    // implemented explicitly in AggQuantile struct
                    unimplemented!()
                },
                GroupByMethod::NanMin => {
                    #[cfg(feature = "propagate_nans")]
                    {
                        let (s, groups) = ac.get_final_aggregation();
                        let agg_s = if s.dtype().is_float() {
                            nan_propagating_aggregate::group_agg_nan_min_s(&s, &groups)
                        } else {
                            s.agg_min(&groups)
                        };
                        AggregatedScalar(rename_series(agg_s, &keep_name))
                    }
                    #[cfg(not(feature = "propagate_nans"))]
                    {
                        panic!("activate 'propagate_nans' feature")
                    }
                },
                GroupByMethod::NanMax => {
                    #[cfg(feature = "propagate_nans")]
                    {
                        let (s, groups) = ac.get_final_aggregation();
                        let agg_s = if s.dtype().is_float() {
                            nan_propagating_aggregate::group_agg_nan_max_s(&s, &groups)
                        } else {
                            s.agg_max(&groups)
                        };
                        AggregatedScalar(rename_series(agg_s, &keep_name))
                    }
                    #[cfg(not(feature = "propagate_nans"))]
                    {
                        panic!("activate 'propagate_nans' feature")
                    }
                },
            }
        };

        Ok(AggregationContext::from_agg_state(
            out,
            Cow::Borrowed(groups),
        ))
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        if let Some(field) = self.field.as_ref() {
            Ok(field.clone())
        } else {
            self.input.to_field(input_schema)
        }
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }
}

fn rename_series(mut s: Series, name: &str) -> Series {
    s.rename(name);
    s
}

impl PartitionedAggregation for AggregationExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        let expr = self.input.as_partitioned_aggregator().unwrap();
        let series = expr.evaluate_partitioned(df, groups, state)?;

        // SAFETY:
        // groups are in bounds
        unsafe {
            match self.agg_type {
                #[cfg(feature = "dtype-struct")]
                GroupByMethod::Mean => {
                    let new_name = series.name().to_string();

                    // ensure we don't overflow
                    // the all 8 and 16 bits integers are already upcasted to int16 on `agg_sum`
                    let mut agg_s = if matches!(series.dtype(), DataType::Int32 | DataType::UInt32)
                    {
                        series.cast(&DataType::Int64).unwrap().agg_sum(groups)
                    } else {
                        series.agg_sum(groups)
                    };
                    agg_s.rename(&new_name);

                    if !agg_s.dtype().is_numeric() {
                        Ok(agg_s)
                    } else {
                        let agg_s = match agg_s.dtype() {
                            DataType::Float32 => agg_s,
                            _ => agg_s.cast(&DataType::Float64).unwrap(),
                        };
                        let mut count_s = series.agg_valid_count(groups);
                        count_s.rename("__POLARS_COUNT");
                        Ok(StructChunked::new(&new_name, &[agg_s, count_s])
                            .unwrap()
                            .into_series())
                    }
                },
                GroupByMethod::Implode => {
                    let new_name = series.name();
                    let mut agg = series.agg_list(groups);
                    agg.rename(new_name);
                    Ok(agg)
                },
                GroupByMethod::First => {
                    let mut agg = series.agg_first(groups);
                    agg.rename(series.name());
                    Ok(agg)
                },
                GroupByMethod::Last => {
                    let mut agg = series.agg_last(groups);
                    agg.rename(series.name());
                    Ok(agg)
                },
                GroupByMethod::Max => {
                    let mut agg = series.agg_max(groups);
                    agg.rename(series.name());
                    Ok(agg)
                },
                GroupByMethod::Min => {
                    let mut agg = series.agg_min(groups);
                    agg.rename(series.name());
                    Ok(agg)
                },
                GroupByMethod::Sum => {
                    let mut agg = series.agg_sum(groups);
                    agg.rename(series.name());
                    Ok(agg)
                },
                GroupByMethod::Count {
                    include_nulls: true,
                } => {
                    let mut ca = groups.group_count();
                    ca.rename(series.name());
                    Ok(ca.into_series())
                },
                _ => {
                    unimplemented!()
                },
            }
        }
    }

    fn finalize(
        &self,
        partitioned: Series,
        groups: &GroupsProxy,
        _state: &ExecutionState,
    ) -> PolarsResult<Series> {
        match self.agg_type {
            GroupByMethod::Count {
                include_nulls: true,
            }
            | GroupByMethod::Sum => {
                let mut agg = unsafe { partitioned.agg_sum(groups) };
                agg.rename(partitioned.name());
                Ok(agg)
            },
            #[cfg(feature = "dtype-struct")]
            GroupByMethod::Mean => {
                let new_name = partitioned.name();
                match partitioned.dtype() {
                    DataType::Struct(_) => {
                        let ca = partitioned.struct_().unwrap();
                        let sum = &ca.fields()[0];
                        let count = &ca.fields()[1];
                        let (agg_count, agg_s) =
                            unsafe { POOL.join(|| count.agg_sum(groups), || sum.agg_sum(groups)) };
                        let agg_s = &agg_s / &agg_count;
                        Ok(rename_series(agg_s, new_name))
                    },
                    _ => Ok(Series::full_null(
                        new_name,
                        groups.len(),
                        partitioned.dtype(),
                    )),
                }
            },
            GroupByMethod::Implode => {
                // the groups are scattered over multiple groups/sub dataframes.
                // we now must collect them into a single group
                let ca = partitioned.list().unwrap();
                let new_name = partitioned.name().to_string();

                let mut values = Vec::with_capacity(groups.len());
                let mut can_fast_explode = true;

                let mut offsets = Vec::<i64>::with_capacity(groups.len() + 1);
                let mut length_so_far = 0i64;
                offsets.push(length_so_far);

                let mut process_group = |ca: ListChunked| -> PolarsResult<()> {
                    let s = ca.explode()?;
                    length_so_far += s.len() as i64;
                    offsets.push(length_so_far);
                    values.push(s.chunks()[0].clone());

                    if s.len() == 0 {
                        can_fast_explode = false;
                    }
                    Ok(())
                };

                match groups {
                    GroupsProxy::Idx(groups) => {
                        for (_, idx) in groups {
                            let ca = unsafe {
                                // SAFETY:
                                // The indexes of the group_by operation are never out of bounds
                                ca.take_unchecked(idx)
                            };
                            process_group(ca)?;
                        }
                    },
                    GroupsProxy::Slice { groups, .. } => {
                        for [first, len] in groups {
                            let len = *len as usize;
                            let ca = ca.slice(*first as i64, len);
                            process_group(ca)?;
                        }
                    },
                }

                let vals = values.iter().map(|arr| &**arr).collect::<Vec<_>>();
                let values = concatenate(&vals).unwrap();

                let data_type = ListArray::<i64>::default_datatype(values.data_type().clone());
                // SAFETY: offsets are monotonically increasing.
                let arr = ListArray::<i64>::new(
                    data_type,
                    unsafe { Offsets::new_unchecked(offsets).into() },
                    values,
                    None,
                );
                let mut ca = ListChunked::with_chunk(&new_name, arr);
                if can_fast_explode {
                    ca.set_fast_explode()
                }
                Ok(ca.into_series().as_list().into_series())
            },
            GroupByMethod::First => {
                let mut agg = unsafe { partitioned.agg_first(groups) };
                agg.rename(partitioned.name());
                Ok(agg)
            },
            GroupByMethod::Last => {
                let mut agg = unsafe { partitioned.agg_last(groups) };
                agg.rename(partitioned.name());
                Ok(agg)
            },
            GroupByMethod::Max => {
                let mut agg = unsafe { partitioned.agg_max(groups) };
                agg.rename(partitioned.name());
                Ok(agg)
            },
            GroupByMethod::Min => {
                let mut agg = unsafe { partitioned.agg_min(groups) };
                agg.rename(partitioned.name());
                Ok(agg)
            },
            _ => unimplemented!(),
        }
    }
}

pub struct AggQuantileExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) quantile: Arc<dyn PhysicalExpr>,
    pub(crate) interpol: QuantileInterpolOptions,
}

impl AggQuantileExpr {
    pub fn new(
        input: Arc<dyn PhysicalExpr>,
        quantile: Arc<dyn PhysicalExpr>,
        interpol: QuantileInterpolOptions,
    ) -> Self {
        Self {
            input,
            quantile,
            interpol,
        }
    }

    fn get_quantile(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<f64> {
        let quantile = self.quantile.evaluate(df, state)?;
        polars_ensure!(quantile.len() <= 1, ComputeError:
            "polars only supports computing a single quantile; \
            make sure the 'quantile' expression input produces a single quantile"
        );
        quantile.get(0).unwrap().try_extract()
    }
}

impl PhysicalExpr for AggQuantileExpr {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    fn get_name(&self) -> &str {
        "AggQuantile"
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let input = self.input.evaluate(df, state)?;
        let quantile = self.get_quantile(df, state)?;
        input.quantile_as_series(quantile, self.interpol)
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;
        // don't change names by aggregations as is done in polars-core
        let keep_name = ac.series().name().to_string();

        let quantile = self.get_quantile(df, state)?;

        // SAFETY:
        // groups are in bounds
        let mut agg = unsafe {
            ac.flat_naive()
                .into_owned()
                .agg_quantile(ac.groups(), quantile, self.interpol)
        };
        agg.rename(&keep_name);
        Ok(AggregationContext::from_agg_state(
            AggregatedScalar(agg),
            Cow::Borrowed(groups),
        ))
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema)
    }
}

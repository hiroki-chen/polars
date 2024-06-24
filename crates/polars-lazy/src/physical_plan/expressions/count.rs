use std::borrow::Cow;

use picachv::expr_argument::Argument;
use picachv::native::build_expr;
use picachv::ExprArgument;
use polars_core::prelude::*;
use polars_plan::constants::LEN;
use uuid::Uuid;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct CountExpr {
    expr: Expr,
    expr_uuid: Uuid,
}

impl CountExpr {
    pub(crate) fn new(ctx_id: Uuid, policy_check: bool) -> PolarsResult<Self> {
        let expr_uuid = if policy_check {
            let expr_arg = ExprArgument {
                argument: Some(Argument::Count(picachv::CountExpr {})),
            };
            build_expr(ctx_id, expr_arg)
                .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?
        } else {
            Uuid::new_v4()
        };

        Ok(Self {
            expr: Expr::Len,
            expr_uuid,
        })
    }
}

impl PhysicalExpr for CountExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn get_uuid(&self) -> Uuid {
        self.expr_uuid
    }

    fn evaluate(&self, df: &DataFrame, _state: &ExecutionState) -> PolarsResult<Series> {
        Ok(Series::new("len", [df.height() as IdxSize]))
    }

    fn get_name(&self) -> &str {
        "Count"
    }

    fn evaluate_on_groups<'a>(
        &self,
        _df: &DataFrame,
        groups: &'a GroupsProxy,
        _state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let ca = groups.group_count().with_name(LEN);
        let s = ca.into_series();
        Ok(AggregationContext::new(s, Cow::Borrowed(groups), true))
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(Field::new(LEN, IDX_DTYPE))
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }
}

impl PartitionedAggregation for CountExpr {
    #[allow(clippy::ptr_arg)]
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        self.evaluate_on_groups(df, groups, state)
            .map(|mut ac| ac.aggregated())
    }

    /// Called to merge all the partitioned results in a final aggregate.
    #[allow(clippy::ptr_arg)]
    fn finalize(
        &self,
        partitioned: Series,
        groups: &GroupsProxy,
        _state: &ExecutionState,
    ) -> PolarsResult<Series> {
        // SAFETY: groups are in bounds.
        let agg = unsafe { partitioned.agg_sum(groups) };
        Ok(agg.with_name(LEN))
    }
}

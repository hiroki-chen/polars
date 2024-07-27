use std::borrow::Cow;

use picachv::column_specifier::Column;
use picachv::native::{build_expr, reify_expression};
use picachv::{ColumnSpecifier, ExprArgument};
use polars_core::prelude::*;
use polars_plan::constants::CSE_REPLACED;
use uuid::Uuid;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct ColumnExpr {
    name: Arc<str>,
    expr: Expr,
    schema: Option<SchemaRef>,
    // The id on the arena.
    expr_id: Uuid,
}

impl ColumnExpr {
    pub fn new(
        name: Arc<str>,
        expr: Expr,
        schema: Option<SchemaRef>,
        ctx_id: Uuid,
        policy_check: bool,
    ) -> PolarsResult<Self> {
        let expr_id = if policy_check {
            let expr_arg = ExprArgument {
                argument: Some(picachv::expr_argument::Argument::Column(
                    picachv::ColumnExpr {
                        column: Some(ColumnSpecifier {
                            column: Some(Column::ColumnName(name.to_string())),
                        }),
                    },
                )),
            };

            build_expr(ctx_id, expr_arg)
                .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?
        } else {
            Default::default()
        };

        Ok(Self {
            name,
            expr,
            schema,
            expr_id,
        })
    }
}

impl ColumnExpr {
    fn check_external_context(
        &self,
        out: PolarsResult<Series>,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        match out {
            Ok(col) => Ok(col),
            Err(e) => {
                if state.ext_contexts.is_empty() {
                    Err(e)
                } else {
                    for df in state.ext_contexts.as_ref() {
                        let out = df.column(&self.name);
                        if out.is_ok() {
                            return out.cloned();
                        }
                    }
                    Err(e)
                }
            },
        }
    }

    fn process_by_idx(
        &self,
        out: &Series,
        _state: &ExecutionState,
        _schema: &Schema,
        df: &DataFrame,
        check_state_schema: bool,
    ) -> PolarsResult<Series> {
        if out.name() != &*self.name {
            if check_state_schema {
                if let Some(schema) = _state.get_schema() {
                    return self.process_from_state_schema(df, _state, &schema);
                }
            }

            // this path should not happen
            #[cfg(feature = "panic_on_schema")]
            {
                if _state.ext_contexts.is_empty()
                    && std::env::var("POLARS_NO_SCHEMA_CHECK").is_err()
                {
                    panic!(
                        "got {} expected: {} from schema: {:?} and DataFrame: {:?}",
                        out.name(),
                        &*self.name,
                        _schema,
                        df
                    )
                }
            }
            // in release we fallback to linear search
            #[allow(unreachable_code)]
            {
                df.column(&self.name).cloned()
            }
        } else {
            Ok(out.clone())
        }
    }
    fn process_by_linear_search(
        &self,
        df: &DataFrame,
        _state: &ExecutionState,
        _panic_during_test: bool,
    ) -> PolarsResult<Series> {
        #[cfg(feature = "panic_on_schema")]
        {
            if _panic_during_test
                && _state.ext_contexts.is_empty()
                && std::env::var("POLARS_NO_SCHEMA_CHECK").is_err()
            {
                panic!("invalid schema: df {:?};\ncolumn: {}", df, &self.name)
            }
        }
        // in release we fallback to linear search
        #[allow(unreachable_code)]
        df.column(&self.name).cloned()
    }

    fn process_from_state_schema(
        &self,
        df: &DataFrame,
        state: &ExecutionState,
        schema: &Schema,
    ) -> PolarsResult<Series> {
        match schema.get_full(&self.name) {
            None => self.process_by_linear_search(df, state, true),
            Some((idx, _, _)) => match df.get_columns().get(idx) {
                Some(out) => self.process_by_idx(out, state, schema, df, false),
                None => self.process_by_linear_search(df, state, true),
            },
        }
    }

    fn process_cse(&self, df: &DataFrame, schema: &Schema) -> PolarsResult<(usize, Series)> {
        // The CSE columns are added on the rhs.
        let offset = schema.len();
        let columns = &df.get_columns()[offset..];
        // Linear search will be relatively cheap as we only search the CSE columns.
        Ok(columns
            .iter()
            .enumerate()
            .find(|(_, s)| s.name() == self.name.as_ref())
            .map(|(idx, s)| (idx, s.clone()))
            .unwrap())
    }
}

impl PhysicalExpr for ColumnExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn get_uuid(&self) -> Uuid {
        self.expr_id
    }

    fn get_name(&self) -> &str {
        "Column"
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        // Try to get its name!
        let out = match &self.schema {
            None => self.process_by_linear_search(df, state, false),
            Some(schema) => {
                match schema.get_full(&self.name) {
                    Some((idx, _, _)) => {
                        // check if the schema was correct
                        // if not do O(n) search
                        match df.get_columns().get(idx) {
                            Some(out) => {
                                if state.policy_check {
                                    let idx = idx.to_le_bytes();
                                    reify_expression(state.ctx_id, self.expr_id, &idx).map_err(
                                        |e| PolarsError::ComputeError(e.to_string().into()),
                                    )?;
                                }

                                self.process_by_idx(out, state, schema, df, true)
                            },
                            None => {
                                // partitioned group_by special case
                                if let Some(schema) = state.get_schema() {
                                    self.process_from_state_schema(df, state, &schema)
                                } else {
                                    self.process_by_linear_search(df, state, true)
                                }
                            },
                        }
                    },
                    // in the future we will throw an error here
                    // now we do a linear search first as the lazy reported schema may still be incorrect
                    // in debug builds we panic so that it can be fixed when occurring
                    None => {
                        if self.name.starts_with(CSE_REPLACED) {
                            let (idx, s) = self.process_cse(df, schema)?;
                            if state.policy_check {
                                let idx = idx.to_le_bytes();
                                reify_expression(state.ctx_id, self.expr_id, &idx)
                                    .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
                            }

                            return Ok(s);
                        }
                        self.process_by_linear_search(df, state, true)
                    },
                }
            },
        };
        self.check_external_context(out, state)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let s = self.evaluate(df, state)?;
        Ok(AggregationContext::new(s, Cow::Borrowed(groups), false))
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        input_schema.get_field(&self.name).ok_or_else(|| {
            polars_err!(
                ColumnNotFound: "could not find {:?} in schema: {:?}", self.name, &input_schema
            )
        })
    }
}

impl PartitionedAggregation for ColumnExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        _groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        self.evaluate(df, state)
    }

    fn finalize(
        &self,
        partitioned: Series,
        _groups: &GroupsProxy,
        _state: &ExecutionState,
    ) -> PolarsResult<Series> {
        Ok(partitioned)
    }
}

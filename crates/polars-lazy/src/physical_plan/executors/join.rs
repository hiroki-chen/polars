use picachv::{plan_argument::Argument, transform_info::Information, JoinInformation, PlanArgument, TransformArgument, TransformInfo};
use polars_ops::frame::DataFrameJoinOps;
use uuid::Uuid;

use super::*;

pub struct JoinExec {
    input_left: Option<Box<dyn Executor>>,
    input_right: Option<Box<dyn Executor>>,
    left_on: Vec<Arc<dyn PhysicalExpr>>,
    right_on: Vec<Arc<dyn PhysicalExpr>>,
    parallel: bool,
    args: JoinArgs,
}

impl JoinExec {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input_left: Box<dyn Executor>,
        input_right: Box<dyn Executor>,
        left_on: Vec<Arc<dyn PhysicalExpr>>,
        right_on: Vec<Arc<dyn PhysicalExpr>>,
        parallel: bool,
        args: JoinArgs,
    ) -> Self {
        JoinExec {
            input_left: Some(input_left),
            input_right: Some(input_right),
            left_on,
            right_on,
            parallel,
            args,
        }
    }
}

impl Executor for JoinExec {
    fn execute<'a>(&'a mut self, state: &'a mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run JoinExec")
            }
        }
        if state.verbose() {
            eprintln!("join parallel: {}", self.parallel);
        };
        let mut input_left = self.input_left.take().unwrap();
        let mut input_right = self.input_right.take().unwrap();

        let mut lhs_df_uuid = Uuid::default();
        let mut rhs_df_uuid = Uuid::default();

        let (mut df_left, mut df_right) = if self.parallel {
            let mut state_right = state.split();
            let mut state_left = state.split();
            state_right.branch_idx += 1;
            // propagate the fetch_rows static value to the spawning threads.
            let fetch_rows = FETCH_ROWS.with(|fetch_rows| fetch_rows.get());

            let (df_left, df_right) = POOL.join(
                 || {
                    FETCH_ROWS.with(|fr| fr.set(fetch_rows));
                    input_left.execute(&mut state_left)
                },
                 || {
                    FETCH_ROWS.with(|fr| fr.set(fetch_rows));
                    input_right.execute(&mut state_right)
                },
            );

            let df_left = df_left?;
            let df_right = df_right?;

            lhs_df_uuid = state_left.active_df_uuid;
            rhs_df_uuid = state_right.active_df_uuid;

            (df_left, df_right)
        } else {
            let mut state_right = state.split();
            let mut state_left = state.split();

            let (df_left, df_right) = (input_left.execute(&mut state_left)?, input_right.execute(&mut state_right)?);

            lhs_df_uuid = state_left.active_df_uuid;
            rhs_df_uuid = state_right.active_df_uuid;

            (df_left, df_right)
        };        

        let profile_name = if state.has_node_timer() {
            let by = self
                .left_on
                .iter()
                .map(|s| Ok(s.to_field(&df_left.schema())?.name))
                .collect::<PolarsResult<Vec<_>>>()?;
            let name = comma_delimited("join".to_string(), &by);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        let (df, ti) = state.record(|| {
            let left_on_series = self
                .left_on
                .iter()
                .map(|e| e.evaluate(&df_left, state))
                .collect::<PolarsResult<Vec<_>>>()?;

            let right_on_series = self
                .right_on
                .iter()
                .map(|e| e.evaluate(&df_right, state))
                .collect::<PolarsResult<Vec<_>>>()?;

            // make sure that we can join on evaluated expressions
            for s in &left_on_series {
                df_left.with_column(s.clone())?;
            }
            for s in &right_on_series {
                df_right.with_column(s.clone())?;
            }

            // prepare the tolerance
            // we must ensure that we use the right units
            #[cfg(feature = "asof_join")]
            {
                if let JoinType::AsOf(options) = &mut self.args.how {
                    use polars_core::utils::arrow::temporal_conversions::MILLISECONDS_IN_DAY;
                    if let Some(tol) = &options.tolerance_str {
                        let duration = polars_time::Duration::parse(tol);
                        polars_ensure!(
                            duration.months() == 0,
                            ComputeError: "cannot use month offset in timedelta of an asof join; \
                            consider using 4 weeks"
                        );
                        let left_asof = df_left.column(left_on_series[0].name())?;
                        use DataType::*;
                        match left_asof.dtype() {
                            Datetime(tu, _) | Duration(tu) => {
                                let tolerance = match tu {
                                    TimeUnit::Nanoseconds => duration.duration_ns(),
                                    TimeUnit::Microseconds => duration.duration_us(),
                                    TimeUnit::Milliseconds => duration.duration_ms(),
                                };
                                options.tolerance = Some(AnyValue::from(tolerance))
                            }
                            Date => {
                                let days = (duration.duration_ms() / MILLISECONDS_IN_DAY) as i32;
                                options.tolerance = Some(AnyValue::from(days))
                            }
                            Time => {
                                let tolerance = duration.duration_ns();
                                options.tolerance = Some(AnyValue::from(tolerance))
                            }
                            _ => {
                                panic!("can only use timedelta string language with Date/Datetime/Duration/Time dtypes")
                            }
                        }
                    }
                }
            }

            let mut ti = JoinInformation::default();
            ti.lhs_df_uuid = lhs_df_uuid.to_bytes_le().to_vec();
            ti.rhs_df_uuid = rhs_df_uuid.to_bytes_le().to_vec();

            let df = df_left._join_impl(
                &df_right,
                left_on_series,
                right_on_series,
                self.args.clone(),
                true,
                state.verbose(),
                &mut ti
            );

            if state.verbose() {
                eprintln!("{:?} join dataframes finished", self.args.how);
            };
            
            match df {
                Ok(df) => Ok((df, ti)),
                Err(e) => {
                    if state.verbose() {
                        eprintln!("{:?}", e)
                    }
                    return Err(e);
                }
            }

        }, profile_name)?;

        if state.policy_check {
            let arg = PlanArgument {
                argument: Some(Argument::Transform(TransformArgument {})),
                transform_info: Some(TransformInfo { information: Some(
                    Information::Join(ti)
                ) }),
            };
            
            self.execute_epilogue(state, Some(arg))?;
        }

        Ok(df)
    }
}

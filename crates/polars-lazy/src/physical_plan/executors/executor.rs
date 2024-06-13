use picachv::native::execute_epilogue;
use picachv::PlanArgument;

use super::*;

// Executor are the executors of the physical plan and produce DataFrames. They
// combine physical expressions, which produce Series.

/// Executors will evaluate physical expressions and collect them in a DataFrame.
///
/// Executors have other executors as input. By having a tree of executors we can execute the
/// physical plan until the last executor is evaluated.
pub trait Executor: Send {
    fn execute(&mut self, cache: &mut ExecutionState) -> PolarsResult<DataFrame>;

    fn execute_epilogue(
        &self,
        cache: &mut ExecutionState,
        plan_arg: Option<PlanArgument>,
    ) -> PolarsResult<()> {
        println!(
            "executing epilogue: {} {} {:?}",
            cache.ctx_id,
            cache.active_df_uuid,
            plan_arg,
        );

        let active_df_uuid = execute_epilogue(cache.ctx_id, cache.active_df_uuid, plan_arg)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
        println!("setting to {active_df_uuid}");
        cache.set_active_df_uuid(active_df_uuid);
        cache.transform.take();

        Ok(())
    }
}

pub struct Dummy {}
impl Executor for Dummy {
    fn execute(&mut self, _cache: &mut ExecutionState) -> PolarsResult<DataFrame> {
        panic!("should not get here");
    }
}

impl Default for Box<dyn Executor> {
    fn default() -> Self {
        Box::new(Dummy {})
    }
}

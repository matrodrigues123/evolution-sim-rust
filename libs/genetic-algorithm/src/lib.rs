use rand::RngCore;

pub struct GeneticAlgorithm;

pub trait Individual {
    fn fitness(&self) -> f32;
}
pub trait SelectionMethod {
    fn select<'a, I: Individual>(
        &self, 
        rng: &mut dyn RngCore,
        population: &'a [I]) -> &'a I;
}

impl GeneticAlgorithm {
    pub fn evolve<I>(&self, population: &[I]) -> Vec<I> {
        assert!(!population.is_empty());
        (0.. population.len())
            .map(|_| {
                // todo selection
                // todo crossover
                // todo mutation
                todo!()
            })
            .collect()
    }
}
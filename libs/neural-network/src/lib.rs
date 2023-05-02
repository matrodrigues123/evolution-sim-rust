use rand::{Rng, RngCore};

pub struct Network {
    layers: Vec<Layer>,
}
pub struct LayerTopology {
    pub neurons: usize,
}
impl Network {
    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopology]) -> Self {
        // let mut built_layers = Vec::new();
        // for i in 0..(layers.len() - 1) {
        //     let input_neurons = layers[i].neurons;
        //     let output_neurons = layers[i + 1].neurons;

        //     built_layers.push(Layer::random(
        //         input_neurons,
        //         output_neurons,
        //     ))
        // }

        // rust has a patter for iterating through adjacent elements
        let built_layers = layers
            .windows(2)
            .map(|layers| {
                Layer::random(rng, layers[0].neurons, layers[1].neurons)
            })
            .collect();

        Self { layers: built_layers }
    }
    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        // this is the folding pattern:
        
        // for layer in &self.layers {
        //     inputs = layer.propagate(inputs);
        // }
        //inputs

        // how to do it with functional programming
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(&inputs))
    }
}
    

struct Layer {
    neurons: Vec<Neuron>,
}
impl Layer {
    fn random(rng: &mut dyn RngCore, input_neurons: usize, output_neurons: usize) -> Self {
        // output_neurons here is the size of the output 
        // of the previous layer
        let neurons = (0..output_neurons)
            .map(|_| Neuron::random(rng, input_neurons))
            .collect();

        Self { neurons }
    }

    fn propagate(&self, inputs: &[f32]) -> Vec<f32> {
        // can use Vec::new(), but this is inefficient
        // we can prealocate our vector with the size of neurons

        // let mut outputs = Vec::with_capacity(self.neurons.len());

        // for neuron in &self.neurons {
        //     let output = neuron.propagate(&inputs);
        //     outputs.push(output);
        // }
        // outputs

        // functional programming implementation
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(inputs))
            .collect()

    }
}


struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}
impl Neuron {
    fn random(rng: &mut dyn rand::RngCore, weights_size: usize) -> Self {
        let bias = rng.gen_range(-1.0..=1.0);

        let weights = (0..weights_size)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();

        Self { bias, weights }
    }
    fn propagate(&self, inputs: &[f32]) -> f32{
        assert_eq!(inputs.len(), self.weights.len());

        // let mut output = 0.0;

        // for (i, input) in inputs.iter().enumerate() {
        //     output += input*self.weights[i];
        // }
        // output += self.bias;
        
        // the loop can be simplified with zip, 
        // since they have the same lenght, this makes sense
        // for (input, weight) in inputs.iter().zip(&self.weights)  {
        //     output += input * weight;
        // }

        // simplify even further by using map
        let output: f32 = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input*weight)
            .sum();
                                
        // add bias and apply activation function
        (self.bias + output).max(0.0) // returns maximum between 0 and output
    }
}

// Neuron tests
#[cfg(test)]
mod tests {
    use super::*;

    mod random {
        use super::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use approx::assert_relative_eq;

        #[test]
        fn test_neuron_random() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let neuron = Neuron::random(&mut rng, 4);

            assert_relative_eq!(neuron.bias, -0.6255188);
            assert_relative_eq!(neuron.weights.as_slice(), &[0.67383957, 0.8181262, 0.26284897, 0.5238807].as_ref());
        }

        #[test]
        fn test_layer_random() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let layer = Layer::random(&mut rng, 3, 2);

            assert_eq!(layer.neurons.len(), 2);
            assert_eq!(layer.neurons[0].weights.len(), 3);
            assert_eq!(layer.neurons[1].weights.len(), 3);

            assert_relative_eq!(layer.neurons[0].bias, -0.6255188);
            assert_relative_eq!(layer.neurons[1].bias, 0.5238807);

            assert_relative_eq!(layer.neurons[0].weights.as_slice(), &[0.67383957, 0.8181262, 0.26284897].as_ref());
            assert_relative_eq!(layer.neurons[1].weights.as_slice(), &[-0.53516835, 0.069369674, -0.7648182].as_ref());
        }

        #[test]
        fn test_network_random() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let network = Network::random(&mut rng, &[LayerTopology {
                neurons: 3,
            }, LayerTopology {
                neurons: 2,
            }, LayerTopology {
                neurons: 1,
            }]);

            // layers are defined as 2 by 2 pairs
            assert_eq!(network.layers.len(), 2);

            // for the first pair 3 - 2
            // there are two output neurons
            assert_eq!(network.layers[0].neurons.len(), 2);
            // but each receive 3 inputs (from 'previous' layer), 
            // so their weight lenghts is equal to 3
            assert_eq!(network.layers[0].neurons[0].weights.len(), 3);
            // assert the values
            assert_relative_eq!(network.layers[0].neurons[0].bias, -0.6255188);
            assert_relative_eq!(network.layers[0].neurons[0].weights.as_slice(), &[0.67383957, 0.8181262, 0.26284897].as_ref());
            assert_relative_eq!(network.layers[0].neurons[1].bias, 0.5238807);
            assert_relative_eq!(network.layers[0].neurons[1].weights.as_slice(), &[-0.53516835, 0.069369674, -0.7648182].as_ref());

            // for the second pair 2 - 1
            // there is one output neuron
            assert_eq!(network.layers[1].neurons.len(), 1);
            // but it receive 2 inputs (from actual previous layer),
            // so its weight length is equal to 2
            assert_eq!(network.layers[1].neurons[0].weights.len(), 2);
            // assert the values
            assert_relative_eq!(network.layers[1].neurons[0].bias, -0.102499366);
            assert_relative_eq!(network.layers[1].neurons[0].weights.as_slice(), &[-0.48879617, -0.19277132].as_ref());
            
        }


    }

    mod propagate {
        use approx::assert_relative_eq;

        use super::*;

        #[test]
        fn test_neuron_propagate() {
            let neuron = Neuron {
                bias: 0.5,
                weights: vec![-0.3, 0.8],
            };
            
            // testes .max(), the ReLU
            assert_relative_eq!(neuron.propagate(&[-10.0, -10.0]), 0.0);
            // compare to one example by hand
            assert_relative_eq!(neuron.propagate(&[0.5, 1.0]), (-0.3 * 0.5) + (0.8*1.0) + 0.5);
        }

        #[test]
        fn test_layer_propagate() {
            let neurons = vec![
                Neuron {
                    bias: 0.5,
                    weights: vec![-0.3, 0.8, 0.5],
                },
            ];
            let layer = Layer {
                neurons
            };
            
            assert_eq!(layer.propagate(&[1.0, 2.0, 3.0]), vec![1.0*(-0.3) + 2.0*0.8 + 3.0*0.5 + 0.5]);
        }

        #[test]
        fn test_network_propagate() {
            let layers = vec![
                Layer {
                    neurons: vec![
                        Neuron {
                            bias: 0.5,
                            weights: vec![-0.3, 0.8, 0.5],
                        },
                    ],
                },
                Layer {
                    neurons: vec![
                        Neuron {
                            bias: -0.2,
                            weights: vec![0.1],
                        },
                        Neuron {
                            bias: 0.4,
                            weights: vec![0.3],
                        },
                    ],
                },
            ];
            let network = Network {
                layers,
            };

            let first_prop = 1.0 * (-0.3) + 2.0 * (0.8) + 3.0*(0.5) + 0.5;
            let second_prop1 = first_prop*0.1 - 0.2;
            let second_prop2 = first_prop*0.3 + 0.4;
            assert_eq!(network.propagate(vec![1.0, 2.0, 3.0]), vec![second_prop1, second_prop2]);
            
        }

    }
}
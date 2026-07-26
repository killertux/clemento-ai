use itertools::Itertools;
use rand::random;
use std::{
    f32,
    ops::{Add, SubAssign},
};

use ndarray::{Array1, Array2, Axis, Ix2, arr1, arr2};

pub struct Mlp {
    weights: Vec<Array2<f32>>,
    biases: Vec<Array1<f32>>,
}

impl Mlp {
    pub fn new(topology: &[usize]) -> Self {
        topology.iter().tuple_windows().fold(
            Self {
                weights: Vec::new(),
                biases: Vec::new(),
            },
            |mut acc, element: (&usize, &usize)| {
                let mut value = Array2::zeros(Ix2(*element.0, *element.1));
                value.iter_mut().for_each(|v| *v = random());
                acc.weights.push(value);
                let mut value = Array1::zeros(*element.1);
                value.iter_mut().for_each(|v| *v = random());
                acc.biases.push(value);
                acc
            },
        )
    }

    pub fn forward(&self, x: Array1<f32>) -> Array1<f32> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .fold(x, |acc, element| {
                acc.dot(element.0)
                    .add(element.1)
                    .mapv(|v| 1.0 / (1.0 + (-v).exp()))
            })
    }

    pub fn forward_with_cache(&self, x: Array1<f32>) -> Vec<Array1<f32>> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .fold(vec![x], |mut acc, element| {
                let z = acc[acc.len() - 1]
                    .dot(element.0)
                    .add(element.1)
                    .mapv(|v| 1.0 / (1.0 + (-v).exp()));
                acc.push(z);
                acc
            })
    }

    pub fn backpropagate(
        &self,
        x: Array1<f32>,
        target: Array1<f32>,
    ) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        let cache = self.forward_with_cache(x);
        let l = self.weights.len();

        let mut d_weights: Vec<Array2<f32>> = vec![Array2::zeros((0, 0)); l];
        let mut d_biases: Vec<Array1<f32>> = vec![Array1::zeros(0); l];

        let mut delta = &cache[l] - &target;

        for layer in (0..l).rev() {
            let a_prev = &cache[layer];

            d_weights[layer] = a_prev
                .view()
                .insert_axis(Axis(1))
                .dot(&delta.view().insert_axis(Axis(0)));
            d_biases[layer] = delta.clone(); // ← dL/db = δ

            if layer > 0 {
                let routed = self.weights[layer].dot(&delta);
                let slope = a_prev.mapv(|a| a * (1.0 - a));
                delta = routed * slope;
            }
        }
        (d_weights, d_biases)
    }

    pub fn train(&mut self, x: Array1<f32>, target: Array1<f32>, lr: f32) {
        let (d_weights, d_biases) = self.backpropagate(x, target);
        self.weights
            .iter_mut()
            .zip(d_weights.iter())
            .for_each(|(weight, d_weight)| weight.sub_assign(&(d_weight * lr)));
        self.biases
            .iter_mut()
            .zip(d_biases.iter())
            .for_each(|(bias, d_bias)| bias.sub_assign(&(d_bias * lr)));
    }
}

static AND_X: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
static AND_Y: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
static XOR_X: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
static XOR_Y: [f32; 4] = [0.0, 1.0, 1.0, 0.0];
static OR_X: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
static OR_Y: [f32; 4] = [0.0, 1.0, 1.0, 1.0];
static NAND_X: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
static NAND_Y: [f32; 4] = [1.0, 1.0, 1.0, 0.0];

fn main() {
    let mut mlp = Mlp::new(&[2, 2, 1]);
    (0..20000).into_iter().for_each(|_| {
        XOR_X
            .iter()
            .zip(XOR_Y.iter())
            .for_each(|(x, t)| mlp.train(arr1(x), arr1(&[*t]), 0.1))
    });

    println!("x1|  x2|  y");
    for x in XOR_X {
        let y = mlp.forward(arr1(&x));
        println!("{}|{}|{}", x[0], x[1], y[0])
    }
    // let mlp = MLP {
    //     output: [Neuron {
    //         weights: [10.67, 10.67],
    //         bias: -16.1,
    //     }],
    //     hidden_layer: [[
    //         Neuron {
    //             weights: [11.9, 11.9],
    //             bias: -5.5,
    //         },
    //         Neuron {
    //             weights: [-10.6, -10.6],
    //             bias: 16.1,
    //         },
    //     ]],
    // };

    // println!("x1   |x2    |y");
    // let x = [0.0, 0.0];
    // let y = mlp.forward(&x);
    // println!("0.00 | 0.00 | {:.2}", y.0[0]);
    // let x = [1.0, 0.0];
    // let y = mlp.forward(&x);
    // println!("1.00 | 0.00 | {:.2}", y.0[0]);
    // let x = [0.0, 1.0];
    // let y = mlp.forward(&x);
    // println!("0.00 | 1.00 | {:.2}", y.0[0]);
    // let x = [1.0, 1.0];
    // let y = mlp.forward(&x);
    // println!("1.00 | 1.00 | {:.2}", y.0[0]);

    // let mut neuron = Neuron {
    //     weights: [0.5, 0.5],
    //     bias: 0.0,
    // };
    // let inputs = AND_X;
    // let targets = AND_Y;

    // let steps = 10000;
    // let chunk = 500;
    // for _ in 0..(steps / chunk) {
    //     println!("Errors: {}", mean_square_error(&neuron, &inputs, &targets));
    //     train(
    //         &mut neuron,
    //         &inputs,
    //         &targets,
    //         gradient_binary_cross_entry,
    //         0.5,
    //         chunk,
    //     )
    // }
    // println!("Errors: {}", mean_square_error(&neuron, &inputs, &targets));
    // println!("{:?}", neuron);
}

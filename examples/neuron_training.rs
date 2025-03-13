/*
Neuron Training with RayBNN

This script trains a neural network using RayBNN and exports key training data.
*/

extern crate arrayfire;
extern crate raybnn;

use std::time::{Duration, Instant};
use raybnn::physics::update_f32::add_neuron_option_type;
use raybnn::physics::initial_f32;
use raybnn::graph::large_sparse_i32;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;

#[allow(unused_must_use)]
fn main() {
    // 初始化 GPU
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);
    arrayfire::set_seed(1231);

    // 设定神经网络参数
    let dir_path = "/tmp/".to_string();
    let max_input_size: u64 = 10;
    let input_size: u64 = 10;
    let max_output_size: u64 = 1;
    let output_size: u64 = 1;
    let max_neuron_size: u64 = 500;
    let batch_size: u64 = 32;
    let traj_size = 1;

    // 创建初始神经网络结构
    let mut arch_search = raybnn::interface::automatic_f32::create_start_archtecture(
        input_size,
        max_input_size,
        output_size,
        max_output_size,
        max_neuron_size,
        batch_size,
        traj_size,
        &dir_path,
    );

    // 添加神经元
    let add_neuron_options: add_neuron_option_type = add_neuron_option_type {
        new_active_size: 50,  // 添加 50 个神经元
        init_connection_num: 500,
        input_neuron_con_rad: 40.0 * arch_search.neural_network.netdata.neuron_rad,
        hidden_neuron_con_rad: 40.0 * arch_search.neural_network.netdata.neuron_rad,
        output_neuron_con_rad: 40.0 * arch_search.neural_network.netdata.neuron_rad,
    };

    raybnn::physics::update_f32::add_neuron_to_existing3(&add_neuron_options, &mut arch_search);

    // 训练神经网络
    println!("Training neural network...");
    let start_time = Instant::now();
    
    for epoch in 0..100 {  // 训练 100 轮
        raybnn::neural::network_f32::state_space_forward_batch(
            &arch_search.neural_network.netdata,
            &arch_search.neural_network.neuron_pos,
            &arch_search.neural_network.WRowIdxCSR,
            &arch_search.neural_network.WColIdx,
            &arch_search.neural_network.network_params,
        );

        // 计算损失
        let loss = raybnn::optimal::loss_f32::MSE(&arch_search.neural_network.neuron_pos, &arch_search.neural_network.neuron_pos);
        println!("Epoch {}: Loss = {}", epoch, loss);

        if loss < 0.0001 {
            break;
        }
    }
    let elapsed_time = start_time.elapsed();
    println!("Training completed in {:.2} seconds", elapsed_time.as_secs_f64());

    // 导出神经网络数据
    let filename = "./neuron_training_results.csv";
    raybnn::export::dataloader_f32::write_arr_to_csv(
        &filename,
        &arch_search.neural_network.neuron_pos,
    );

    println!("Training results saved to {}", filename);
}

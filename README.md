# Qwen 2.5 7B VL Fine-tuning for Latex OCR

A comprehensive setup for efficiently fine-tuning Qwen 2 7B Vision-Language model using state-of-the-art optimization techniques and memory-efficient training methods. This project specifically fine-tunes the model using the **unsloth/Latex_OCR** dataset to improve LaTeX generation capabilities from mathematical images.

## About

This repository provides an optimized environment for fine-tuning the Qwen 2.5 7B Vision-Language model, leveraging cutting-edge techniques for memory efficiency and training acceleration. The setup combines parameter-efficient fine-tuning methods (LoRA), quantization techniques, and optimized attention mechanisms to enable training large vision-language models on consumer hardware.

**Project Focus**: This implementation fine-tunes the model on the **unsloth/Latex_OCR** dataset to enhance the model's ability to generate accurate LaTeX code from mathematical expressions and formulas in images.

### Key Features

- **Memory Efficient**: Utilizes 4-bit/8-bit quantization and LoRA for reduced memory footprint
- **Accelerated Training**: Optimized attention mechanisms and custom GPU kernels
- **Vision-Language Support**: Full support for multimodal training with images and text
- **Production Ready**: Includes RLHF capabilities and distributed training support
- **Easy to Use**: Streamlined setup with pre-configured optimizations

### LoRA Diagram
```
LoRA Architecture for Qwen 2.5 7B VL:

Original Weight Matrix W (frozen)          LoRA Update Path
┌─────────────────────────────────┐       ┌─────────┐    ┌─────────┐
│                                 │       │    A    │    │    B    │
│                                 │       │ 16×3584 │    │ 3584×16 │
│         Frozen Original         │       │         │    │         │
│      Attention Weights          │       │ Rank 16 │    │ Rank 16 │
│        (3584×3584)              │       │         │    │         │
│     12.8M parameters            │       └─────────┘    └─────────┘
│                                 │             │              │
└─────────────────────────────────┘             │              │
              │                                 │              │
              │                                 └──────┬───────┘
              │                                        │
              │                                   LoRA Update
              │                                        │
              └────────────────┬───────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Next Layer       │
                    │   (W + ΔW = W + BA) │
                    └─────────────────────┘

Qwen 2.5 7B VL Specifications:
- Hidden dimension: 3584 (inherited from base [Qwen 2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B) architecture)
- Attention heads: 28 for Q, 4 for KV (Grouped Query Attention)
- Layers: 28
- Total parameters: 8.29B (VL model, from [official VL model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct))
- Additional components: Vision Transformer + multimodal fusion layers

LoRA Parameter Reduction (per attention layer):
- Original attention weights (4 matrices): 
  * q_proj: 3584×3584 = 12.8M parameters
  * k_proj: 3584×3584 = 12.8M parameters  
  * v_proj: 3584×3584 = 12.8M parameters
  * o_proj: 3584×3584 = 12.8M parameters
  * Total per layer: 51.2M parameters

- LoRA weights (4 matrix pairs):
  * Each A matrix: 16×3584 = 57,344 parameters
  * Each B matrix: 3584×16 = 57,344 parameters  
  * Per projection: 114,688 parameters
  * Total per layer: 458,752 parameters

- Memory reduction: 99.1% per attention layer~
- Across 28 layers: 1.43B → 12.8M trainable parameters
```

*LoRA keeps the original weight matrix W frozen and learns two small low-rank matrices A and B. The final output combines both paths: original weights + low-rank adaptation.*

## Installation

### Prerequisites

- Google Colab with A100 GPU runtime
- Python 3.8+
- PyTorch with CUDA support (pre-installed in Colab)

### Quick Setup

```bash
# Core optimization libraries (installed without dependencies for version control)
pip install --no-deps bitsandbytes accelerate xformers==0.029.post3 peft trl triton cut_cross_entropy unsloth_zoo

# Data processing and model hub libraries
pip install sentencepiece protobuf datasets huggingface_hub hf_transfer

# Unsloth optimization framework
pip install --no-deps unsloth
```

## Dependencies Explained

### Core Optimization Libraries

#### **bitsandbytes**
- **Purpose**: 8-bit and 4-bit neural network quantization
- **Benefits**: Reduces memory usage by 50-75% while maintaining model performance
- **Use Case**: Enables training large models on consumer GPUs

#### **accelerate**
- **Purpose**: Hugging Face's distributed training and mixed precision library
- **Benefits**: Automatic device placement, gradient accumulation, multi-GPU support
- **Use Case**: Simplifies scaling training across multiple devices

#### **xformers** (v0.029.post3)
- **Purpose**: Memory-efficient attention mechanisms and transformer optimizations
- **Benefits**: Significantly faster attention computation with reduced memory usage
- **Use Case**: Accelerates transformer model training and inference

#### **peft** (Parameter-Efficient Fine-Tuning)
- **Purpose**: Implements LoRA, AdaLoRA, and other efficient fine-tuning methods
- **Benefits**: Fine-tune only 0.1-1% of model parameters while achieving full fine-tuning performance
- **Use Case**: Enables fine-tuning large models with minimal computational resources

#### **trl** (Transformer Reinforcement Learning)
- **Purpose**: Reinforcement learning for language models
- **Benefits**: Implements PPO, RLHF, and other RL techniques for LLMs
- **Use Case**: Advanced training techniques for alignment and instruction following

#### **triton**
- **Purpose**: GPU kernel programming language
- **Benefits**: Write custom CUDA operations in Python-like syntax
- **Use Case**: Custom optimized operations for specific model architectures

#### **cut_cross_entropy**
- **Purpose**: Optimized cross-entropy loss implementation
- **Benefits**: More memory and compute efficient than standard implementations
- **Use Case**: Faster training with reduced memory overhead

#### **unsloth_zoo**
- **Purpose**: Pre-configured model setups and optimizations
- **Benefits**: Ready-to-use configurations for popular models
- **Use Case**: Quick setup for common fine-tuning scenarios

### Data Processing Libraries

#### **sentencepiece**
- **Purpose**: Google's subword tokenization library
- **Benefits**: Implements BPE and unigram tokenization used by modern LLMs
- **Use Case**: Text preprocessing and tokenization for Qwen models

#### **protobuf**
- **Purpose**: Google's Protocol Buffers for data serialization
- **Benefits**: Efficient serialization format used by ML frameworks
- **Use Case**: Model and data serialization/deserialization

#### **datasets**
- **Purpose**: Hugging Face's dataset processing library
- **Benefits**: Unified interface to thousands of datasets with built-in preprocessing
- **Use Case**: Loading and preprocessing training data

#### **huggingface_hub**
- **Purpose**: Client library for Hugging Face Hub
- **Benefits**: Easy model/dataset download and upload
- **Use Case**: Accessing pre-trained models and sharing fine-tuned models

#### **hf_transfer**
- **Purpose**: Faster download mechanism for Hugging Face Hub
- **Benefits**: Rust-based implementation for improved download speeds
- **Use Case**: Faster model and dataset downloads

### Optimization Framework

#### **unsloth**
- **Purpose**: Comprehensive LLM fine-tuning optimization framework
- **Benefits**: 2-5x faster training with 50% less memory usage
- **Use Case**: Primary optimization layer for efficient Qwen 2.5 fine-tuning

## Performance Benefits

- **Memory Reduction**: Up to 75% less VRAM usage through quantization and LoRA
- **Speed Improvement**: 2-5x faster training with optimized attention and kernels
- **A100 Optimization**: Fully utilizes A100's 40GB VRAM for efficient 7B model training
- **Quality Preservation**: Maintain model performance while using efficient techniques

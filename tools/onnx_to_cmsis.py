#!/usr/bin/env python3
"""Convert INT8 ONNX model to CMSIS-NN compatible C code for STM32."""

import argparse
import os
import numpy as np
import onnx
from onnx import numpy_helper


class OnnxToCMSISConverter:
    """Convert INT8 ONNX to CMSIS-NN C code."""
    
    def __init__(self, onnx_path: str, output_path: str, calibration_data: np.ndarray = None):
        self.onnx_path = onnx_path
        self.output_path = output_path
        self.calibration_data = calibration_data
        
    def load_model(self):
        """Load ONNX model."""
        return onnx.load(self.onnx_path)
    
    def extract_weights(self, model):
        """Extract weights and biases from model."""
        weights = {}
        for initializer in model.graph.initializer:
            name = initializer.name
            array = numpy_helper.to_array(initializer)
            weights[name] = array
        return weights
    
    def compute_quantization_params(self, data, symmetric=True):
        """Compute scale and zero point for quantization."""
        if symmetric:
            abs_max = max(abs(data.min()), abs(data.max()))
            scale = abs_max / 127.0
            zero_point = 0
        else:
            data_min, data_max = data.min(), data.max()
            scale = (data_max - data_min) / 255.0
            zero_point = int(-data_min / scale)
        return scale, zero_point
    
    def quantize_weights(self, weights, scale, zero_point=0):
        """Quantize weights to int8."""
        quantized = np.round(weights / scale + zero_point).astype(np.int8)
        return np.clip(quantized, -128, 127)
    
    def generate_c_header(self, model, input_dim, output_dim, layer_dims):
        """Generate C header file."""
        header = f"""/* Auto-generated from {os.path.basename(self.onnx_path)} */
#ifndef POLICY_INT8_H
#define POLICY_INT8_H

#include <stdint.h>
#include "arm_nnfunctions.h"

/* Model dimensions */
#define POLICY_INPUT_DIM {input_dim}
#define POLICY_OUTPUT_DIM {output_dim}
#define POLICY_HIDDEN1_DIM {layer_dims[0]}
#define POLICY_HIDDEN2_DIM {layer_dims[1]}

/* Buffer sizes for CMSIS-NN */
#define POLICY_BUFFER_SIZE (2 * POLICY_HIDDEN1_DIM)

/* Function prototypes */
void policy_init(void);
void policy_inference_int8(const float* obs_float, float* actions_float);

#endif /* POLICY_INT8_H */
"""
        return header
    
    def generate_c_source(self, model, weights, calibration_data):
        """Generate C source file with CMSIS-NN implementation."""
        
        # Parse model structure
        layers = self._parse_layers(model, weights)
        
        # Compute quantization parameters
        if calibration_data is not None:
            input_scale, _ = self.compute_quantization_params(calibration_data)
        else:
            input_scale = 0.1  # Default scale
        
        # Generate code
        source = f"""/* Auto-generated CMSIS-NN inference code */
#include "policy_int8.h"
#include <math.h>
#include <string.h>

/* Input quantization parameters */
static const float INPUT_SCALE = {input_scale:.9f}f;
static const float OUTPUT_SCALE = 1.0f;  /* Assuming normalized outputs */

/* Quantization parameters per layer */
"""
        
        # Generate weight arrays and quantization params
        for i, layer in enumerate(layers):
            w = layer['weights']
            b = layer['biases']
            
            w_scale, _ = self.compute_quantization_params(w)
            b_scale, _ = self.compute_quantization_params(b)
            
            w_q = self.quantize_weights(w, w_scale)
            b_q = self.quantize_weights(b, b_scale)
            
            # Flatten weights in row-major order
            w_flat = w_q.T.flatten()
            
            source += f"""
/* Layer {i} weights and biases */
static const q7_t layer{i}_weights[{w_flat.size}] = {{
"""
            # Write weights in rows of 16
            for j in range(0, len(w_flat), 16):
                chunk = w_flat[j:j+16]
                source += "    " + ", ".join(f"{int(v)}" for v in chunk) + ",\n"
            
            source += "};\n\n"
            
            source += f"static const q7_t layer{i}_biases[{len(b_q)}] = {{\n"
            source += "    " + ", ".join(f"{int(v)}" for v in b_q) + "\n};\n\n"
            
            source += f"static const float layer{i}_weight_scale = {w_scale:.9f}f;\n"
            source += f"static const float layer{i}_bias_scale = {b_scale:.9f}f;\n\n"
        
        # Generate inference function
        source += """
/* Scratch buffer for intermediate computations */
static q15_t scratch_buffer[POLICY_BUFFER_SIZE];

void policy_init(void) {
    /* No initialization required */
}

/* Tanh approximation using lookup table (CMSIS-NN style) */
static q7_t tanh_q7(q7_t x) {
    /* Simple linear approximation for demo */
    /* In production, use proper CMSIS-NN tanh_q7 or lookup table */
    return x;  /* Placeholder */
}

void policy_inference_int8(const float* obs_float, float* actions_float) {
    /* Input quantization */
    q7_t input_q7[POLICY_INPUT_DIM];
    for (int i = 0; i < POLICY_INPUT_DIM; i++) {
        float val = obs_float[i] / INPUT_SCALE;
        val = fmaxf(-127.0f, fminf(127.0f, val));
        input_q7[i] = (q7_t)roundf(val);
    }
    
    /* Layer 1: Input -> Hidden1 */
    q7_t hidden1[POLICY_HIDDEN1_DIM];
    arm_fully_connected_q7(
        input_q7,
        layer0_weights,
        POLICY_INPUT_DIM,
        POLICY_HIDDEN1_DIM,
        1, 7,  /* bias_shift, out_shift (tuned) */
        layer0_biases,
        hidden1,
        scratch_buffer
    );
    
    /* Apply Tanh activation */
    for (int i = 0; i < POLICY_HIDDEN1_DIM; i++) {
        hidden1[i] = tanh_q7(hidden1[i]);
    }
    
    /* Layer 2: Hidden1 -> Hidden2 */
    q7_t hidden2[POLICY_HIDDEN2_DIM];
    arm_fully_connected_q7(
        hidden1,
        layer1_weights,
        POLICY_HIDDEN1_DIM,
        POLICY_HIDDEN2_DIM,
        1, 7,
        layer1_biases,
        hidden2,
        scratch_buffer
    );
    
    /* Apply Tanh activation */
    for (int i = 0; i < POLICY_HIDDEN2_DIM; i++) {
        hidden2[i] = tanh_q7(hidden2[i]);
    }
    
    /* Layer 3: Hidden2 -> Output */
    q7_t output_q7[POLICY_OUTPUT_DIM];
    arm_fully_connected_q7(
        hidden2,
        layer2_weights,
        POLICY_HIDDEN2_DIM,
        POLICY_OUTPUT_DIM,
        1, 7,
        layer2_biases,
        output_q7,
        scratch_buffer
    );
    
    /* Dequantize outputs */
    for (int i = 0; i < POLICY_OUTPUT_DIM; i++) {
        actions_float[i] = ((float)output_q7[i]) * OUTPUT_SCALE;
        /* Clamp to [-1, 1] */
        if (actions_float[i] > 1.0f) actions_float[i] = 1.0f;
        if (actions_float[i] < -1.0f) actions_float[i] = -1.0f;
    }
}
"""
        return source
    
    def _parse_layers(self, model, weights):
        """Parse layer structure from ONNX model."""
        layers = []
        
        # Simple parser for sequential MLP
        # Assumes structure: FC -> Tanh -> FC -> Tanh -> FC
        weight_keys = sorted([k for k in weights.keys() if 'weight' in k.lower()])
        bias_keys = sorted([k for k in weights.keys() if 'bias' in k.lower()])
        
        for w_key, b_key in zip(weight_keys, bias_keys):
            layers.append({
                'weights': weights[w_key],
                'biases': weights[b_key]
            })
        
        return layers
    
    def convert(self):
        """Run conversion process."""
        print(f"Loading ONNX model: {self.onnx_path}")
        model = self.load_model()
        
        print("Extracting weights...")
        weights = self.extract_weights(model)
        
        # Get dimensions
        input_dim = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
        output_dim = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
        
        # Infer hidden dimensions from weights
        layers = self._parse_layers(model, weights)
        layer_dims = [layers[0]['weights'].shape[1], layers[1]['weights'].shape[1]]
        
        print(f"Model structure: {input_dim} -> {layer_dims[0]} -> {layer_dims[1]} -> {output_dim}")
        
        # Generate header
        header_path = self.output_path.replace('.c', '.h')
        header = self.generate_c_header(model, input_dim, output_dim, layer_dims)
        with open(header_path, 'w') as f:
            f.write(header)
        print(f"Generated header: {header_path}")
        
        # Generate source
        source = self.generate_c_source(model, weights, self.calibration_data)
        with open(self.output_path, 'w') as f:
            f.write(source)
        print(f"Generated source: {self.output_path}")
        
        # Report statistics
        total_params = sum(w['weights'].size + w['biases'].size for w in layers)
        print(f"Total parameters: {total_params:,}")
        print(f"Estimated Flash: {total_params} bytes (~{total_params/1024:.1f} KB)")
        print(f"Estimated RAM: ~{2*layer_dims[0]} bytes (scratch buffer)")


def main():
    parser = argparse.ArgumentParser(description="Convert INT8 ONNX to CMSIS-NN C code")
    parser.add_argument("--onnx", required=True, help="Path to INT8 ONNX model")
    parser.add_argument("--output", required=True, help="Output C file path")
    parser.add_argument("--calibration", help="Path to calibration data (.npy)")
    
    args = parser.parse_args()
    
    # Load calibration data if provided
    calibration_data = None
    if args.calibration and os.path.exists(args.calibration):
        calibration_data = np.load(args.calibration)
        print(f"Loaded calibration data: shape {calibration_data.shape}")
    
    # Convert
    converter = OnnxToCMSISConverter(args.onnx, args.output, calibration_data)
    converter.convert()
    
    print("\n[SUCCESS] Conversion complete!")
    print(f"Generated files: {args.output}, {args.output.replace('.c', '.h')}")


if __name__ == "__main__":
    main()

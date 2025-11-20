#!/usr/bin/env python3
"""
ONNX to C Converter with INT8 Quantization Support for STM32 Deployment

Converts ONNX models (FP32 or INT8 quantized) to plain C code for embedded systems.
Supports MLP networks with common activations (ReLU, Tanh, Sigmoid).

Usage:
    # For FP32 models:
    python onnx_to_c_converter.py --onnx policy.onnx --output policy_fp32.c --dtype float32

    # For INT8 models (recommended for STM32):
    python onnx_to_c_converter.py --onnx policy_int8.onnx --output policy_int8.c --dtype int8
"""

import argparse
import os
import sys
import numpy as np
import onnx
from onnx import numpy_helper
from typing import Dict, List, Tuple


class OnnxToCConverter:
    """Convert ONNX models to C code for embedded deployment."""

    def __init__(self, onnx_path: str, output_path: str, dtype: str = "float32", namespace: str = "policy"):
        self.onnx_path = onnx_path
        self.output_path = output_path
        self.dtype = dtype
        self.namespace = namespace
        self.use_int8 = dtype == "int8"
        
    def load_model(self) -> onnx.ModelProto:
        """Load and validate ONNX model."""
        try:
            model = onnx.load(self.onnx_path)
            onnx.checker.check_model(model)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def extract_initializers(self, model: onnx.ModelProto) -> Dict[str, np.ndarray]:
        """Extract weight tensors from model."""
        initializers = {}
        for tensor in model.graph.initializer:
            arr = numpy_helper.to_array(tensor)
            initializers[tensor.name] = arr
        return initializers
    
    def analyze_quantization(self, model: onnx.ModelProto) -> Dict:
        """Extract quantization parameters from INT8 model."""
        quant_params = {}
        
        # Look for QuantizeLinear/DequantizeLinear nodes
        for node in model.graph.node:
            if node.op_type == "QuantizeLinear":
                # Extract scale and zero point
                quant_params[node.output[0]] = {
                    "scale": node.input[1] if len(node.input) > 1 else None,
                    "zero_point": node.input[2] if len(node.input) > 2 else None,
                }
            elif node.op_type == "DequantizeLinear":
                quant_params[node.input[0]] = {
                    "scale": node.input[1] if len(node.input) > 1 else None,
                    "zero_point": node.input[2] if len(node.input) > 2 else None,
                }
        
        return quant_params
    
    def generate_c_header(self, model: onnx.ModelProto, initializers: Dict) -> str:
        """Generate C header with model parameters."""
        lines = []
        lines.append(f"/* Auto-generated from {os.path.basename(self.onnx_path)} */\n")
        lines.append(f"#ifndef {self.namespace.upper()}_H\n")
        lines.append(f"#define {self.namespace.upper()}_H\n\n")
        lines.append("#include <stdint.h>\n")
        lines.append("#include <math.h>\n\n")
        
        # Determine input/output dimensions
        input_tensor = model.graph.input[0]
        output_tensor = model.graph.output[0]
        
        input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        output_shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        
        input_dim = input_shape[-1] if input_shape else 1
        output_dim = output_shape[-1] if output_shape else 1
        
        lines.append(f"#define {self.namespace.upper()}_INPUT_DIM {input_dim}\n")
        lines.append(f"#define {self.namespace.upper()}_OUTPUT_DIM {output_dim}\n\n")
        
        # Data type definitions
        if self.use_int8:
            lines.append("typedef int8_t weight_t;\n")
            lines.append("typedef int32_t acc_t;\n")
            lines.append("typedef float scale_t;\n\n")
        else:
            lines.append("typedef float weight_t;\n")
            lines.append("typedef float acc_t;\n\n")
        
        # Function declarations
        lines.append(f"/* Initialize the model (call once) */\n")
        lines.append(f"void {self.namespace}_init(void);\n\n")
        lines.append(f"/* Run inference */\n")
        if self.use_int8:
            lines.append(f"void {self.namespace}_infer(const float* input, float* output);\n\n")
        else:
            lines.append(f"void {self.namespace}_infer(const float* input, float* output);\n\n")
        
        lines.append(f"#endif /* {self.namespace.upper()}_H */\n")
        
        return "".join(lines)
    
    def generate_c_implementation(self, model: onnx.ModelProto, initializers: Dict) -> str:
        """Generate C implementation with inference code."""
        lines = []
        lines.append(f"#include \"{self.namespace}.h\"\n\n")
        
        # Extract layers
        layers = self._extract_layers(model, initializers)
        
        # Generate weight arrays
        for i, layer in enumerate(layers):
            if "weights" in layer:
                W = layer["weights"]
                if self.use_int8:
                    W_quantized = np.clip(np.round(W / layer.get("w_scale", 1.0)), -128, 127).astype(np.int8)
                    lines.append(self._generate_array(f"{self.namespace}_W{i}", W_quantized, "int8_t"))
                    lines.append(f"static const float {self.namespace}_W{i}_scale = {layer.get('w_scale', 1.0):.9f}f;\n\n")
                else:
                    lines.append(self._generate_array(f"{self.namespace}_W{i}", W.astype(np.float32), "float"))
            
            if "bias" in layer and layer["bias"] is not None:
                b = layer["bias"]
                if self.use_int8:
                    b_quantized = np.clip(np.round(b / layer.get("b_scale", 1.0)), -128, 127).astype(np.int8)
                    lines.append(self._generate_array(f"{self.namespace}_b{i}", b_quantized, "int8_t"))
                    lines.append(f"static const float {self.namespace}_b{i}_scale = {layer.get('b_scale', 1.0):.9f}f;\n\n")
                else:
                    lines.append(self._generate_array(f"{self.namespace}_b{i}", b.astype(np.float32), "float"))
        
        # Generate activation functions
        lines.append(self._generate_activations())
        
        # Generate init function
        lines.append(f"void {self.namespace}_init(void) {{\n")
        lines.append("    /* Initialization if needed */\n")
        lines.append("}\n\n")
        
        # Generate inference function
        lines.append(f"void {self.namespace}_infer(const float* input, float* output) {{\n")
        lines.append(self._generate_inference_code(layers))
        lines.append("}\n")
        
        return "".join(lines)
    
    def _extract_layers(self, model: onnx.ModelProto, initializers: Dict) -> List[Dict]:
        """Extract layer information from ONNX graph."""
        layers = []
        
        for node in model.graph.node:
            if node.op_type in ["Gemm", "MatMul"]:
                layer = {"type": "linear"}
                
                # Extract weights
                if node.op_type == "Gemm":
                    weight_name = node.input[1]
                    layer["weights"] = initializers[weight_name]
                    
                    # Check for bias
                    if len(node.input) > 2:
                        bias_name = node.input[2]
                        layer["bias"] = initializers[bias_name]
                    else:
                        layer["bias"] = None
                    
                    # Handle transpose attributes
                    for attr in node.attribute:
                        if attr.name == "transB" and onnx.helper.get_attribute_value(attr) == 1:
                            layer["weights"] = layer["weights"].T
                
                elif node.op_type == "MatMul":
                    weight_name = node.input[1]
                    layer["weights"] = initializers[weight_name]
                    layer["bias"] = None
                
                # Estimate quantization scales for INT8
                if self.use_int8:
                    W = layer["weights"]
                    layer["w_scale"] = np.max(np.abs(W)) / 127.0
                    if layer["bias"] is not None:
                        b = layer["bias"]
                        layer["b_scale"] = np.max(np.abs(b)) / 127.0
                
                layer["activation"] = "linear"
                layers.append(layer)
            
            elif node.op_type in ["Relu", "Tanh", "Sigmoid"]:
                # Attach activation to previous layer
                if layers:
                    layers[-1]["activation"] = node.op_type.lower()
        
        return layers
    
    def _generate_array(self, name: str, arr: np.ndarray, dtype: str) -> str:
        """Generate C array declaration."""
        flat = arr.flatten()
        lines = [f"static const {dtype} {name}[{len(flat)}] = {{\n"]
        
        # Write values in rows of 8
        for i in range(0, len(flat), 8):
            chunk = flat[i:i+8]
            if dtype == "int8_t":
                values = ", ".join(f"{int(v)}" for v in chunk)
            else:
                values = ", ".join(f"{v:.9f}f" for v in chunk)
            lines.append(f"    {values},\n")
        
        lines.append("};\n\n")
        return "".join(lines)
    
    def _generate_activations(self) -> str:
        """Generate activation function implementations."""
        code = """/* Activation functions */
static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float tanh_approx(float x) { return tanhf(x); }

"""
        return code
    
    def _generate_inference_code(self, layers: List[Dict]) -> str:
        """Generate layer-by-layer inference code."""
        lines = []
        lines.append(f"    float temp_in[{self.namespace.upper()}_INPUT_DIM];\n")
        lines.append(f"    float temp_out[256];  /* Adjust size as needed */\n\n")
        
        lines.append("    /* Copy input */\n")
        lines.append(f"    for (int i = 0; i < {self.namespace.upper()}_INPUT_DIM; i++) {{\n")
        lines.append("        temp_in[i] = input[i];\n")
        lines.append("    }\n\n")
        
        for i, layer in enumerate(layers):
            lines.append(f"    /* Layer {i} */\n")
            W_shape = layer["weights"].shape
            in_dim, out_dim = W_shape[0], W_shape[1]
            
            lines.append(f"    for (int j = 0; j < {out_dim}; j++) {{\n")
            if self.use_int8:
                lines.append("        float acc = 0.0f;\n")
                lines.append(f"        for (int k = 0; k < {in_dim}; k++) {{\n")
                lines.append(f"            acc += temp_in[k] * (float){self.namespace}_W{i}[j * {in_dim} + k] * {self.namespace}_W{i}_scale;\n")
                lines.append("        }\n")
                if layer["bias"] is not None:
                    lines.append(f"        acc += (float){self.namespace}_b{i}[j] * {self.namespace}_b{i}_scale;\n")
            else:
                lines.append("        float acc = 0.0f;\n")
                lines.append(f"        for (int k = 0; k < {in_dim}; k++) {{\n")
                lines.append(f"            acc += temp_in[k] * {self.namespace}_W{i}[j * {in_dim} + k];\n")
                lines.append("        }\n")
                if layer["bias"] is not None:
                    lines.append(f"        acc += {self.namespace}_b{i}[j];\n")
            
            # Apply activation
            act = layer.get("activation", "linear")
            if act == "relu":
                lines.append("        temp_out[j] = relu(acc);\n")
            elif act == "sigmoid":
                lines.append("        temp_out[j] = sigmoid(acc);\n")
            elif act == "tanh":
                lines.append("        temp_out[j] = tanh_approx(acc);\n")
            else:
                lines.append("        temp_out[j] = acc;\n")
            
            lines.append("    }\n\n")
            
            # Copy output to input for next layer
            if i < len(layers) - 1:
                lines.append("    /* Copy to next layer */\n")
                lines.append(f"    for (int i = 0; i < {out_dim}; i++) {{\n")
                lines.append("        temp_in[i] = temp_out[i];\n")
                lines.append("    }\n\n")
        
        # Copy final output
        lines.append("    /* Copy final output */\n")
        lines.append(f"    for (int i = 0; i < {self.namespace.upper()}_OUTPUT_DIM; i++) {{\n")
        lines.append("        output[i] = temp_out[i];\n")
        lines.append("    }\n")
        
        return "".join(lines)
    
    def convert(self):
        """Run the conversion process."""
        print(f"Loading ONNX model: {self.onnx_path}")
        model = self.load_model()
        
        print("Extracting model parameters...")
        initializers = self.extract_initializers(model)
        
        if self.use_int8:
            print("Analyzing INT8 quantization parameters...")
        
        print(f"Generating C code ({self.dtype})...")
        
        # Generate header
        header_path = self.output_path.replace(".c", ".h")
        header_code = self.generate_c_header(model, initializers)
        with open(header_path, "w") as f:
            f.write(header_code)
        print(f"Generated header: {header_path}")
        
        # Generate implementation
        impl_code = self.generate_c_implementation(model, initializers)
        with open(self.output_path, "w") as f:
            f.write(impl_code)
        print(f"Generated implementation: {self.output_path}")
        
        # Print statistics
        total_params = sum(arr.size for arr in initializers.values())
        if self.use_int8:
            size_bytes = total_params * 1  # 1 byte per int8
        else:
            size_bytes = total_params * 4  # 4 bytes per float32
        
        print(f"\nModel statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Estimated size: {size_bytes:,} bytes ({size_bytes/1024:.2f} KB)")
        print(f"  Data type: {self.dtype}")
        print(f"\nReady for STM32 deployment!")


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX models to C code for STM32")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model file")
    parser.add_argument("--output", required=True, help="Output C file path")
    parser.add_argument("--dtype", choices=["float32", "int8"], default="float32",
                        help="Data type for weights (int8 recommended for STM32)")
    parser.add_argument("--namespace", default="policy", help="Namespace for generated code")
    
    args = parser.parse_args()
    
    converter = OnnxToCConverter(
        onnx_path=args.onnx,
        output_path=args.output,
        dtype=args.dtype,
        namespace=args.namespace
    )
    
    try:
        converter.convert()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

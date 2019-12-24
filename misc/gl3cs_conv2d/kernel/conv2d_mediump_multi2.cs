#version 310 es
layout(local_size_x = 8, local_size_y = 4, local_size_z = 8) in;
layout(std430) buffer;
precision mediump float;

vec4  Vec4FromHalf(in uvec2 v) { return vec4(unpackHalf2x16(v.x), unpackHalf2x16(v.y)); }
uvec2 Vec4ToHalf  (in vec4  v) { return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw)); }

layout(binding = 1) writeonly buffer B1 { uvec2 data[]; } output_data_0;
layout(binding = 0) readonly  buffer B0 { uvec2 data[]; } input_data_0;
layout(binding = 3) readonly  buffer B3 { uvec2 data[]; } bias;
layout(binding = 2) readonly  buffer B2 { uvec2 data[]; } weights;


uniform float clip;
uniform int input_data_0_h;
uniform int input_data_0_w;
uniform int output_data_0_h;
uniform int output_data_0_w;
uniform int src_depth;
uniform int weights_h;
uniform int weights_w;
uniform int workload_x;
uniform int workload_y;
uniform int workload_z;

void main() {

  ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
  if (gid.x >= workload_x || gid.y >= workload_y || gid.z >= workload_z) {
    return;
  }

  /* convolution */
  highp vec4 value_0 = vec4(0);
  highp vec4 result0 = vec4(0);
  highp vec4 result1 = vec4(0);
  vec4 f;
  for (int l = 0; l < src_depth; ++l) {
    vec4 input0 = Vec4FromHalf(input_data_0.data[gid.x * 2 + 0 + input_data_0_w * (gid.y + input_data_0_h * (l))]);
    vec4 input1 = Vec4FromHalf(input_data_0.data[gid.x * 2 + 1 + input_data_0_w * (gid.y + input_data_0_h * (l))]);

    f = Vec4FromHalf(weights.data[0 + weights_w * (l + weights_h * gid.z)]);
    result0[0] += dot(input0, f);
    result1[0] += dot(input1, f);

    f = Vec4FromHalf(weights.data[1 + weights_w * (l + weights_h * gid.z)]);
    result0[1] += dot(input0, f);
    result1[1] += dot(input1, f);

    f = Vec4FromHalf(weights.data[2 + weights_w * (l + weights_h * gid.z)]);
    result0[2] += dot(input0, f);
    result1[2] += dot(input1, f);

    f = Vec4FromHalf(weights.data[3 + weights_w * (l + weights_h * gid.z)]);
    result0[3] += dot(input0, f);
    result1[3] += dot(input1, f);
  }

  /* add bias */
  vec4 b = Vec4FromHalf(bias.data[gid.z]);
  result0 += b;
  result1 += b;

  /* Relu6 */
  result0 = clamp(result0, vec4(0.0), vec4(clip));
  output_data_0.data[gid.x * 2 + 0 + output_data_0_w * (gid.y + output_data_0_h * (gid.z))] = Vec4ToHalf(result0);

  result1 = clamp(result1, vec4(0.0), vec4(clip));
  output_data_0.data[gid.x * 2 + 1 + output_data_0_w * (gid.y + output_data_0_h * (gid.z))] = Vec4ToHalf(result1);
}


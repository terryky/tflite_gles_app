/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "trt_common.h"
#include "trt_posenet.h"
#include <unistd.h>
#include <float.h>

#define UFF_MODEL_PATH      "./models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.uff"
#define PLAN_MODEL_PATH     "./models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.plan"

static Logger               gLogger;
static IExecutionContext   *s_trt_context;
static trt_tensor_t         s_tensor_input;
static trt_tensor_t         s_tensor_heatmap;
static trt_tensor_t         s_tensor_offsets;
static trt_tensor_t         s_tensor_fw_disp;
static trt_tensor_t         s_tensor_bw_disp;
static std::vector<void *>  s_gpu_buffers;

static int     s_img_w = 0;
static int     s_img_h = 0;
static int     s_hmp_w = 0;
static int     s_hmp_h = 0;
static int     s_edge_num = 0;

typedef struct part_score_t {
    float score;
    int   idx_x;
    int   idx_y;
    int   key_id;
} part_score_t;

typedef struct keypoint_t {
    float pos_x;
    float pos_y;
    float score;
    int   valid;
} keypoint_t;


static int pose_edges[][2] =
{
    /* parent,        child */
    { kNose,          kLeftEye      },  //  0
    { kLeftEye,       kLeftEar      },  //  1
    { kNose,          kRightEye     },  //  2
    { kRightEye,      kRightEar     },  //  3
    { kNose,          kLeftShoulder },  //  4
    { kLeftShoulder,  kLeftElbow    },  //  5
    { kLeftElbow,     kLeftWrist    },  //  6
    { kLeftShoulder,  kLeftHip      },  //  7
    { kLeftHip,       kLeftKnee     },  //  8
    { kLeftKnee,      kLeftAnkle    },  //  9
    { kNose,          kRightShoulder},  // 10
    { kRightShoulder, kRightElbow   },  // 11
    { kRightElbow,    kRightWrist   },  // 12
    { kRightShoulder, kRightHip     },  // 13
    { kRightHip,      kRightKnee    },  // 14
    { kRightKnee,     kRightAnkle   },  // 15
};


/* -------------------------------------------------- *
 *  wait counter thread
 * -------------------------------------------------- */
static pthread_t s_wait_thread;
static int       s_is_waiting = 1;

static void *
waitcounter_thread_main (void *)
{
    int cnt = 0;
    while (s_is_waiting)
    {
        fprintf (stderr, "\r%d", cnt++);
        sleep(1);
    }
    return NULL;
}


static int
start_waitcounter_thread ()
{
    s_is_waiting = 1;
    pthread_create (&s_wait_thread, NULL, waitcounter_thread_main, NULL);
    return 0;
}

static int
stop_waitcounter_thread ()
{
    s_is_waiting = 0;
    return 0;
}


/* -------------------------------------------------- *
 *  allocate tensor buffer
 * -------------------------------------------------- */
static int
get_element_count (const Dims &dims)
{
    int elem_count = 1;
    for (int i = 0; i < dims.nbDims; i++)
        elem_count *= dims.d[i];

    return elem_count;
}


static unsigned int
get_element_size (DataType t)
{
    switch (t)
    {
    case DataType::kINT32: return 4;
    case DataType::kFLOAT: return 4;
    case DataType::kHALF:  return 2;
    case DataType::kINT8:  return 1;
    default:               return 0;
    }
}


static void
print_dims (const Dims &dims)
{
    for (int i = 0; i < dims.nbDims; i++)
    {
        if (i > 0)
            fprintf (stderr, "x");

        fprintf (stderr, "%d", dims.d[i]);
    }
}

static const char *
get_type_str (DataType t)
{
    switch (t)
    {
    case DataType::kINT32: return "kINT32";
    case DataType::kFLOAT: return "kFLOAT";
    case DataType::kHALF:  return "kHALF";
    case DataType::kINT8:  return "kINT8";
    default:               return "UNKNOWN";
    }
}


void *
safeCudaMalloc (size_t memSize)
{
    void* deviceMem;
    cudaError_t err;

    err = cudaMalloc(&deviceMem, memSize);
    if (err != 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }

    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}


/* -------------------------------------------------- *
 *  save/load "plan file"
 * -------------------------------------------------- */
static int
emit_plan_file (ICudaEngine *engine, const std::string &plan_file_name)
{
    IHostMemory *serialized_engine = engine->serialize();

    std::ofstream planfile;
    planfile.open (plan_file_name);
    planfile.write((char *)serialized_engine->data(), serialized_engine->size());
    planfile.close();

    return 0;
}


static ICudaEngine *
load_plan_file (const std::string &plan_file_name)
{
    std::ifstream planfile (plan_file_name);
    if (!planfile.is_open())
    {
        fprintf (stderr, "ERR:%s(%d) Could not open plan file.\n", __FILE__, __LINE__);
        return NULL;
    }

    std::stringstream planbuffer;
    planbuffer << planfile.rdbuf();
    std::string plan = planbuffer.str();

    IRuntime *runtime = createInferRuntime (gLogger);

    ICudaEngine *engine;
    engine = runtime->deserializeCudaEngine ((void*)plan.data(), plan.size(), nullptr);

    return engine;
}


/* -------------------------------------------------- *
 *  create cuda engine
 * -------------------------------------------------- */
#define MAX_WORKSPACE (1 << 30)

ICudaEngine *
create_engine_from_uff (const std::string &uff_file)
{
    IBuilder           *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetwork();

    const std::string input_layer_name  = "sub_2";
    const std::string output_layer_name0 = "MobilenetV1/heatmap_2/BiasAdd";
    const std::string output_layer_name1 = "MobilenetV1/offset_2/BiasAdd";
    const std::string output_layer_name2 = "MobilenetV1/displacement_bwd_2/BiasAdd";
    const std::string output_layer_name3 = "MobilenetV1/displacement_fwd_2/BiasAdd";

    IUffParser *parser = nvuffparser::createUffParser();
    parser->registerInput (input_layer_name.c_str(),
                           nvinfer1::Dims3(257, 257, 3), 
                           nvuffparser::UffInputOrder::kNHWC);
    parser->registerOutput(output_layer_name0.c_str());
    parser->registerOutput(output_layer_name1.c_str());
    parser->registerOutput(output_layer_name2.c_str());
    parser->registerOutput(output_layer_name3.c_str());

#if 1
    if (!parser->parse(uff_file.c_str(), *network, nvinfer1::DataType::kHALF))
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }
    builder->setFp16Mode(true);
#else
    if (!parser->parse(uff_file.c_str(), *network, nvinfer1::DataType::kFLOAT))
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }
#endif

    // builder->setInt8Mode (true);
    // builder->setInt8Calibrator (NULL);
    fprintf (stderr, " - HasFastFp16(): %d\n", builder->platformHasFastFp16());
    fprintf (stderr, " - HasFastInt8(): %d\n", builder->platformHasFastInt8());

    /* create the engine */
    builder->setMaxBatchSize (1);
    builder->setMaxWorkspaceSize (MAX_WORKSPACE);

    fprintf (stderr, "Building CUDA Engine. please wait...\n");
    start_waitcounter_thread();

    ICudaEngine *engine = builder->buildCudaEngine (*network);
    if (!engine)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }

    stop_waitcounter_thread();

    network->destroy();
    builder->destroy();
    parser->destroy();

    return engine;
}


int
convert_uff_to_plan (const std::string &plan_file_name, const std::string &uff_file_name)
{
    ICudaEngine *engine;

    engine = create_engine_from_uff (uff_file_name);
    if (!engine)
    {
        fprintf (stderr, "ERR:%s(%d): Failed to load graph from file.\n", __FILE__, __LINE__);
        return -1;
    }

    emit_plan_file (engine, plan_file_name);

    engine->destroy();

    return 0;
}



/* -------------------------------------------------- *
 *  Create TensorRT Interpreter
 * -------------------------------------------------- */
int
trt_get_tensor_by_name (ICudaEngine *engine, const char *name, trt_tensor_t *ptensor)
{
    memset (ptensor, 0, sizeof (*ptensor));

    int bind_idx = -1;
    bind_idx = engine->getBindingIndex (name);
    if (bind_idx < 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
    
    ptensor->bind_idx = bind_idx;
    ptensor->dtype    = engine->getBindingDataType   (bind_idx);
    ptensor->dims     = engine->getBindingDimensions (bind_idx);

    int element_size  = get_element_size  (ptensor->dtype);
    int element_count = get_element_count (ptensor->dims);
    ptensor->memsize  = element_size * element_count;

    ptensor->gpu_mem = safeCudaMalloc(ptensor->memsize);
    ptensor->cpu_mem = malloc (ptensor->memsize);

    fprintf (stderr, "------------------------------\n");
    fprintf (stderr, "[%s]\n", name);
    fprintf (stderr, " bind_idx = %d\n", ptensor->bind_idx);
    fprintf (stderr, " data type= %s\n", get_type_str (ptensor->dtype));
    fprintf (stderr, " dimension= "); print_dims (ptensor->dims);
    fprintf (stderr, "\n");
    fprintf (stderr, "------------------------------\n");
    
    return 0;
}

int
init_trt_posenet ()
{
    fprintf (stderr, "TensorRT version: %d.%d.%d.%d\n",
        NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, NV_TENSORRT_BUILD);

    /* Build TensorRT Engine */
    if (0)
    {
        convert_uff_to_plan (PLAN_MODEL_PATH, UFF_MODEL_PATH);
    }

    /* Load Prebuilt TensorRT Engine */
    fprintf (stderr, "loading cuda engine...\n");
    ICudaEngine *engine = load_plan_file (PLAN_MODEL_PATH);

    s_trt_context = engine->createExecutionContext();


    /* Allocate IO tensors */
    trt_get_tensor_by_name (engine, "sub_2",                                  &s_tensor_input);
    trt_get_tensor_by_name (engine, "MobilenetV1/heatmap_2/BiasAdd",          &s_tensor_heatmap);
    trt_get_tensor_by_name (engine, "MobilenetV1/offset_2/BiasAdd",           &s_tensor_offsets);
    trt_get_tensor_by_name (engine, "MobilenetV1/displacement_fwd_2/BiasAdd", &s_tensor_fw_disp);
    trt_get_tensor_by_name (engine, "MobilenetV1/displacement_bwd_2/BiasAdd", &s_tensor_bw_disp);


    int num_bindings = engine->getNbBindings();
    s_gpu_buffers.resize (num_bindings);
    s_gpu_buffers[s_tensor_input .bind_idx]  = s_tensor_input .gpu_mem;
    s_gpu_buffers[s_tensor_heatmap.bind_idx] = s_tensor_heatmap.gpu_mem;
    s_gpu_buffers[s_tensor_offsets.bind_idx] = s_tensor_offsets.gpu_mem;
    s_gpu_buffers[s_tensor_fw_disp.bind_idx] = s_tensor_fw_disp.gpu_mem;
    s_gpu_buffers[s_tensor_bw_disp.bind_idx] = s_tensor_bw_disp.gpu_mem;

    /* input image dimention */
    s_img_w = s_tensor_input.dims.d[1];
    s_img_h = s_tensor_input.dims.d[0];
    fprintf (stderr, "input image size: (%d, %d)\n", s_img_w, s_img_h);

    /* heatmap dimention */
    s_hmp_w = s_tensor_heatmap.dims.d[1];
    s_hmp_h = s_tensor_heatmap.dims.d[0];
    fprintf (stderr, "heatmap size: (%d, %d)\n", s_hmp_w, s_hmp_h);

    /* displacement forward vector dimention */
    s_edge_num = s_tensor_fw_disp.dims.d[2] / 2;

    return 0;
}

void *
get_posenet_input_buf (int *w, int *h)
{
    *w = s_tensor_input.dims.d[1];
    *h = s_tensor_input.dims.d[0];
    return s_tensor_input.cpu_mem;
}

static float
get_heatmap_score (int idx_y, int idx_x, int key_id)
{
    int idx = (idx_y * s_hmp_w * kPoseKeyNum) + (idx_x * kPoseKeyNum) + key_id;
    float *heatmap_ptr = (float *)s_tensor_heatmap.cpu_mem;
    return heatmap_ptr[idx];
}

static void
get_displacement_vector (void *disp_buf, float *dis_x, float *dis_y, int idx_y, int idx_x, int edge_id)
{
    int idx0 = (idx_y * s_hmp_w * s_edge_num*2) + (idx_x * s_edge_num*2) + (edge_id + s_edge_num);
    int idx1 = (idx_y * s_hmp_w * s_edge_num*2) + (idx_x * s_edge_num*2) + (edge_id);

    float *disp_buf_fp = (float *)disp_buf;
    *dis_x = disp_buf_fp[idx0];
    *dis_y = disp_buf_fp[idx1];
}

static void
get_offset_vector (float *ofst_x, float *ofst_y, int idx_y, int idx_x, int pose_id)
{
    int idx0 = (idx_y * s_hmp_w * kPoseKeyNum*2) + (idx_x * kPoseKeyNum*2) + (pose_id + kPoseKeyNum);
    int idx1 = (idx_y * s_hmp_w * kPoseKeyNum*2) + (idx_x * kPoseKeyNum*2) + (pose_id);
    float *offsets_ptr = (float *)s_tensor_offsets.cpu_mem;

    *ofst_x = offsets_ptr[idx0];
    *ofst_y = offsets_ptr[idx1];
}

/* enqueue an item in descending order. */
static void
enqueue_score (std::list<part_score_t> &queue, int x, int y, int key, float score)
{
    std::list<part_score_t>::iterator itr;
    for (itr = queue.begin(); itr != queue.end(); itr++)
    {
        if (itr->score < score)
            break;
    }

    part_score_t item;
    item.score = score;
    item.idx_x = x;
    item.idx_y = y;
    item.key_id= key;
    queue.insert(itr, item);
}

/*
 * If the score is the highest in local window, return true.
 *
 *    xs    xe
 *   +--+--+--+
 *   |  |  |  | ys
 *   +--+--+--+
 *   |  |##|  |         ##: (idx_x, idx_y)
 *   +--+--+--+
 *   |  |  |  | ye
 *   +--+--+--+
 */
static bool
score_is_max_in_local_window (int key, float score, int idx_y, int idx_x, int max_rad)
{
    int xs = std::max (idx_x - max_rad,     0);
    int ys = std::max (idx_y - max_rad,     0);
    int xe = std::min (idx_x + max_rad + 1, s_hmp_w);
    int ye = std::min (idx_y + max_rad + 1, s_hmp_h);

    for (int y = ys; y < ye; y ++)
    {
        for (int x = xs; x < xe; x ++)
        {
            /* if a higher score is found, return false */
            if (get_heatmap_score (y, x, key) > score)
                return false;
        }
    }
    return true;
}

static void
build_score_queue (std::list<part_score_t> &queue, float thresh, int max_rad)
{
    for (int y = 0; y < s_hmp_h; y ++)
    {
        for (int x = 0; x < s_hmp_w; x ++)
        {
            for (int key = 0; key < kPoseKeyNum; key ++)
            {
                float score = get_heatmap_score (y, x, key);

                /* if this score is lower than thresh, skip this pixel. */
                if (score < thresh)
                    continue;

                /* if there is a higher score near this pixel, skip this pixel. */
                if (!score_is_max_in_local_window (key, score, y, x, max_rad))
                    continue;

                enqueue_score (queue, x, y, key, score);
            }
        }
    }
}

/*
 *  0      28.5    57.1    85.6   114.2   142.7   171.3   199.9   228.4   257   [pos_x]
 *  |---+---|---+---|---+---|---+---|---+---|---+---|---+---|---+---|---+---|
 *     0.0     1.0     2.0     3.0     4.0     5.0     6.0     7.0     8.0      [hmp_pos_x]
 */
static void
get_pos_to_near_index (float pos_x, float pos_y, int *idx_x, int *idx_y)
{
    float ratio_x = pos_x / (float)s_img_w;
    float ratio_y = pos_y / (float)s_img_h;

    float hmp_pos_x = ratio_x * (s_hmp_w - 1);
    float hmp_pos_y = ratio_y * (s_hmp_h - 1);

    int hmp_idx_x = roundf (hmp_pos_x);
    int hmp_idx_y = roundf (hmp_pos_y);

    hmp_idx_x = std::min (hmp_idx_x, s_hmp_w -1);
    hmp_idx_y = std::min (hmp_idx_y, s_hmp_h -1);
    hmp_idx_x = std::max (hmp_idx_x, 0);
    hmp_idx_y = std::max (hmp_idx_y, 0);

    *idx_x = hmp_idx_x;
    *idx_y = hmp_idx_y;
}

static void
get_index_to_pos (int idx_x, int idx_y, int key_id, float *pos_x, float *pos_y)
{
    float ofst_x, ofst_y;
    get_offset_vector (&ofst_x, &ofst_y, idx_y, idx_x, key_id);

    float rel_x = (float)idx_x / (float)(s_hmp_w -1);
    float rel_y = (float)idx_y / (float)(s_hmp_h -1);

    float pos0_x = rel_x * s_img_w;
    float pos0_y = rel_y * s_img_h;

    *pos_x = pos0_x + ofst_x;
    *pos_y = pos0_y + ofst_y;
}


static keypoint_t
traverse_to_tgt_key(int edge, keypoint_t src_key, int tgt_key_id, void *disp)
{
    float src_pos_x = src_key.pos_x;
    float src_pos_y = src_key.pos_y;

    int src_idx_x, src_idx_y;
    get_pos_to_near_index (src_pos_x, src_pos_y, &src_idx_x, &src_idx_y);

    /* get displacement vector from source to target */
    float disp_x, disp_y;
    get_displacement_vector (disp, &disp_x, &disp_y, src_idx_y, src_idx_x, edge);

    /* calculate target position */
    float tgt_pos_x = src_pos_x + disp_x;
    float tgt_pos_y = src_pos_y + disp_y;

    int tgt_idx_x, tgt_idx_y;
    int offset_refine_step = 2;
    for (int i = 0; i < offset_refine_step; i ++)
    {
        get_pos_to_near_index (tgt_pos_x, tgt_pos_y, &tgt_idx_x, &tgt_idx_y);
        get_index_to_pos (tgt_idx_x, tgt_idx_y, tgt_key_id, &tgt_pos_x, &tgt_pos_y);
    }

    keypoint_t tgt_key = {0};
    tgt_key.pos_x = tgt_pos_x;
    tgt_key.pos_y = tgt_pos_y;
    tgt_key.score = get_heatmap_score (tgt_idx_y, tgt_idx_x, tgt_key_id);
    tgt_key.valid = 1;

    return tgt_key;
}

static void
decode_pose (part_score_t &root, keypoint_t *keys)
{
    /* calculate root key position. */
    int idx_x = root.idx_x;
    int idx_y = root.idx_y;
    int keyid = root.key_id;
    float *fw_disp_ptr = (float *)s_tensor_fw_disp.cpu_mem;
    float *bw_disp_ptr = (float *)s_tensor_bw_disp.cpu_mem;

    float pos_x, pos_y;
    get_index_to_pos (idx_x, idx_y, keyid, &pos_x, &pos_y);

    keys[keyid].pos_x = pos_x;
    keys[keyid].pos_y = pos_y;
    keys[keyid].score = root.score;
    keys[keyid].valid = 1;

    for (int edge = s_edge_num - 1; edge >= 0; edge --)
    {
        int src_key_id = pose_edges[edge][1];
        int tgt_key_id = pose_edges[edge][0];

        if ( keys[src_key_id].valid &&
            !keys[tgt_key_id].valid)
        {
            keys[tgt_key_id] = traverse_to_tgt_key(edge, keys[src_key_id], tgt_key_id, bw_disp_ptr);
        }
    }

    for (int edge = 0; edge < s_edge_num; edge ++)
    {
        int src_key_id = pose_edges[edge][0];
        int tgt_key_id = pose_edges[edge][1];

        if ( keys[src_key_id].valid &&
            !keys[tgt_key_id].valid)
        {
            keys[tgt_key_id] = traverse_to_tgt_key(edge, keys[src_key_id], tgt_key_id, fw_disp_ptr);
        }
    }
}

static bool
within_nms_of_corresponding_point (posenet_result_t *pose_result,
                        float pos_x, float pos_y, int key_id, float nms_rad)
{
    for (int i = 0; i < pose_result->num; i ++)
    {
        pose_t *pose = &pose_result->pose[i];
        float prev_pos_x = pose->key[key_id].x * s_img_w;
        float prev_pos_y = pose->key[key_id].y * s_img_h;

        float dx = pos_x - prev_pos_x;
        float dy = pos_y - prev_pos_y;
        float len = (dx * dx) + (dy * dy);

        if (len <= (nms_rad * nms_rad))
            return true;
    }
    return false;
}

static float
get_instance_score (posenet_result_t *pose_result, keypoint_t *keys, float nms_rad)
{
    float score_total = 0.0f;
    for (int i = 0; i < kPoseKeyNum; i ++)
    {
        float pos_x = keys[i].pos_x;
        float pos_y = keys[i].pos_y;
        if (within_nms_of_corresponding_point (pose_result, pos_x, pos_y, i, nms_rad))
            continue;

        score_total += keys[i].score;
    }
    return score_total / (float)kPoseKeyNum;
}

static int
regist_detected_pose (posenet_result_t *pose_result, keypoint_t *keys, float score)
{
    int pose_id = pose_result->num;
    if (pose_id >= MAX_POSE_NUM)
    {
        fprintf (stderr, "ERR: %s(%d): pose_num overflow.\n", __FILE__, __LINE__);
        return -1;
    }

    for (int i = 0; i < kPoseKeyNum; i++)
    {
        pose_result->pose[pose_id].key[i].x     = keys[i].pos_x / (float)s_img_w;
        pose_result->pose[pose_id].key[i].y     = keys[i].pos_y / (float)s_img_h;
        pose_result->pose[pose_id].key[i].score = keys[i].score;
    }

    pose_result->pose[pose_id].pose_score = score;
    pose_result->num ++;

    return 0;
}


static void
decode_multiple_poses (posenet_result_t *pose_result)
{
    std::list<part_score_t> queue;

    float score_thresh  = 0.5f;
    int   local_max_rad = 1;
    build_score_queue (queue, score_thresh, local_max_rad);

    memset (pose_result, 0, sizeof (posenet_result_t));
    while (pose_result->num < MAX_POSE_NUM && !queue.empty())
    {
        part_score_t &root = queue.front();

        float pos_x, pos_y;
        get_index_to_pos (root.idx_x, root.idx_y, root.key_id, &pos_x, &pos_y);

        float nms_rad = 20.0f;
        if (within_nms_of_corresponding_point (pose_result, pos_x, pos_y, root.key_id, nms_rad))
        {
            queue.pop_front();
            continue;
        }

        keypoint_t key_points[kPoseKeyNum] = {0};
        decode_pose (root, key_points);

        float score = get_instance_score (pose_result, key_points, nms_rad);
        regist_detected_pose (pose_result, key_points, score);

        queue.pop_front();
    }
}

static void
decode_single_pose (posenet_result_t *pose_result)
{
    int   max_block_idx[kPoseKeyNum][2] = {0};
    float max_block_cnf[kPoseKeyNum]    = {0};

    /* find the highest heatmap block for each key */
    for (int i = 0; i < kPoseKeyNum; i ++)
    {
        float max_confidence = -FLT_MAX;
        for (int y = 0; y < s_hmp_h; y ++)
        {
            for (int x = 0; x < s_hmp_w; x ++)
            {
                float confidence = get_heatmap_score (y, x, i);
                if (confidence > max_confidence)
                {
                    max_confidence = confidence;
                    max_block_cnf[i] = confidence;
                    max_block_idx[i][0] = x;
                    max_block_idx[i][1] = y;
                }
            }
        }
    }

#if 0
    for (int i = 0; i < kPoseKeyNum; i ++)
    {
        fprintf (stderr, "---------[%d] --------\n", i);
        for (int y = 0; y < s_hmp_h; y ++)
        {
            fprintf (stderr, "[%d] ", y);
            for (int x = 0; x < s_hmp_w; x ++)
            {
                float confidence = get_heatmap_score (y, x, i);
                fprintf (stderr, "%6.3f ", confidence);

                if (x == max_block_idx[i][0] && y == max_block_idx[i][1])
                    fprintf (stderr, "#");
                else
                    fprintf (stderr, " ");
            }
            fprintf (stderr, "\n");
        }
    }
#endif

    /* find the offset vector and calculate the keypoint coordinates. */
    for (int i = 0; i < kPoseKeyNum;i ++ )
    {
        int idx_x = max_block_idx[i][0];
        int idx_y = max_block_idx[i][1];
        float key_posex, key_posey;
        get_index_to_pos (idx_x, idx_y, i, &key_posex, &key_posey);

        pose_result->pose[0].key[i].x     = key_posex / (float)s_img_w;
        pose_result->pose[0].key[i].y     = key_posey / (float)s_img_h;
        pose_result->pose[0].key[i].score = max_block_cnf[i];
    }
    pose_result->num = 1;
    pose_result->pose[0].pose_score = 1.0f;
}


/* -------------------------------------------------- *
 * Invoke TensorRT
 * -------------------------------------------------- */
int
trt_copy_tensor_to_gpu (trt_tensor_t &tensor)
{
    cudaError_t err;

    err = cudaMemcpy (tensor.gpu_mem, tensor.cpu_mem, tensor.memsize, cudaMemcpyHostToDevice);
    if (err != 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
    return 0;
}

int
trt_copy_tensor_from_gpu (trt_tensor_t &tensor)
{
    cudaError_t err;

    err = cudaMemcpy (tensor.cpu_mem, tensor.gpu_mem, tensor.memsize, cudaMemcpyDeviceToHost);
    if (err != 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
    return 0;
}


int
invoke_posenet (posenet_result_t *pose_result)
{
    /* copy to CUDA buffer */
    trt_copy_tensor_to_gpu (s_tensor_input);

    /* invoke inference */
    int batchSize = 1;
    s_trt_context->execute (batchSize, &s_gpu_buffers[0]);

    /* copy from CUDA buffer */
    trt_copy_tensor_from_gpu (s_tensor_heatmap);
    trt_copy_tensor_from_gpu (s_tensor_offsets);
    trt_copy_tensor_from_gpu (s_tensor_fw_disp);
    trt_copy_tensor_from_gpu (s_tensor_bw_disp);

    /*
     * decode algorithm is from:
     *   https://github.com/tensorflow/tfjs-models/tree/master/posenet/src/multi_pose
     */
    if (1)
        decode_multiple_poses (pose_result);
    else
        decode_single_pose (pose_result);

    pose_result->pose[0].heatmap = s_tensor_heatmap.cpu_mem;
    pose_result->pose[0].heatmap_dims[0] = s_hmp_w;
    pose_result->pose[0].heatmap_dims[1] = s_hmp_h;

    return 0;
}


// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net_new.h"

#include "cpu.h"
#include "datareader.h"
#include "layer_type.h"
#include "modelbin.h"
#include "paramdict.h"

#include <stdarg.h>
#include <stdint.h>
#include <string.h>
#include "benchmark.h"
//#include ""

#ifdef _WIN32
#include <algorithm>
#include <windows.h> // Sleep()
#else
#include <unistd.h> // sleep()
#endif

#if NCNN_BENCHMARK
#include "benchmark.h"
#endif // NCNN_BENCHMARK

#if NCNN_VULKAN
#include "command.h"
#include "pipelinecache.h"
#endif // NCNN_VULKAN
#include <ctime>
#include <sys/time.h>
#include <iostream>
#include "datareader.h"
#include "pipeline.h"
#include "arm/file_arm.h"

float pipe_t_list[] = {0.001, 0.235, 0.001, 0.002, 0.001, 0.350, 0.001, 1.787, 0.002, 0.001, 0.002, 0.001, 0.167, 0.001, 0.240, 0.001, 1.894, 0.002, 0.101, 0.001, 0.270, 0.001, 0.003, 0.118, 0.001, 0.003, 0.002, 0.655, 0.002, 0.571, 0.001, 4.697, 0.003, 0.087, 0.001, 1.125, 0.002, 0.002, 0.276, 0.001, 0.002, 0.002, 0.002, 1.710, 0.002, 0.793, 0.001, 3.425, 0.002, 0.136, 0.001, 0.298, 0.001, 0.002, 0.492, 0.001, 0.001, 0.002, 1.282, 0.003, 1.058, 0.002, 2.665, 0.001, 0.130, 0.000, 0.348, 0.001, 0.002, 0.274, 0.001, 0.001, 0.001, 0.503, 0.001, 0.538, 0.001, 2.756, 0.001, 0.155, 0.000, 0.331, 0.000, 0.002, 0.294, 0.000, 0.001, 0.001, 0.527, 0.002, 0.680, 0.001, 3.407, 0.001, 0.164, 0.000, 0.407, 0.001, 0.003, 0.352, 0.002, 0.001, 0.001, 1.261, 0.001, 0.760, 0.002, 4.176, 0.002, 0.146, 0.001, 0.792, 0.001, 0.002, 0.542, 0.001, 0.001, 0.001, 0.001, 1.776, 0.001, 1.108, 0.001, 3.485, 0.002, 0.221, 0.000, 0.779, 0.001, 0.002, 0.902, 0.001, 0.002, 0.002, 2.503, 0.002, 1.309, 0.001, 5.728, 0.002, 0.390, 0.000, 1.087, 0.002, 0.001, 0.836, 0.001, 0.001, 0.001, 0.001, 0.028, 0.002};
std::vector<size_t> DR_file_Vectors;

std::set<int> finish_set;
double save_start_time;

//静态建立一个条件变量infer_cond，放在全局里
pthread_cond_t param_cond = PTHREAD_COND_INITIALIZER;
pthread_cond_t param_cond_1 = PTHREAD_COND_INITIALIZER;


pthread_cond_t param_cond_cpu0 = PTHREAD_COND_INITIALIZER;
pthread_cond_t param_cond_cpu1 = PTHREAD_COND_INITIALIZER;
pthread_cond_t param_cond_cpu2 = PTHREAD_COND_INITIALIZER;
pthread_cond_t param_cond_cpu3 = PTHREAD_COND_INITIALIZER;
pthread_cond_t param_cond_cpu4 = PTHREAD_COND_INITIALIZER;
pthread_cond_t param_cond_cpu5 = PTHREAD_COND_INITIALIZER;
pthread_cond_t param_cond_cpu6 = PTHREAD_COND_INITIALIZER;
pthread_cond_t param_cond_cpu7 = PTHREAD_COND_INITIALIZER;


//静态建立一个互斥锁infer_lock，放在全局里
pthread_mutex_t param_lock = PTHREAD_MUTEX_INITIALIZER;


//静态建立一个条件变量infer_cond，放在全局里
pthread_cond_t infer_cond = PTHREAD_COND_INITIALIZER;
//静态建立一个互斥锁infer_lock，放在全局里
pthread_mutex_t infer_lock = PTHREAD_MUTEX_INITIALIZER;

//静态建立一个条件变量infer_cond，放在全局里
pthread_cond_t pipe_cond = PTHREAD_COND_INITIALIZER;
//静态建立一个互斥锁infer_lock，放在全局里
pthread_mutex_t pipe_lock = PTHREAD_MUTEX_INITIALIZER;


syn_param read_syn = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};
syn_param create_syn = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};
syn_param infer_syn = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};

std::vector<double> read_starts;
std::vector<double> read_ends;
std::vector<double> trans_starts;
std::vector<double> trans_ends;
std::vector<double> infer_starts;
std::vector<double> infer_ends;
int save_time = 0;
void clear_times_save(){
    std::vector<double>().swap(read_starts);
    std::vector<double>().swap(read_ends);
    std::vector<double>().swap(trans_starts);
    std::vector<double>().swap(trans_ends);
    std::vector<double>().swap(infer_starts);
    std::vector<double>().swap(infer_ends);
    save_time = 1;
}
void resize_times_save(int sz){
    read_starts.resize(sz);
    read_ends.resize(sz);
    trans_starts.resize(sz);
    trans_ends.resize(sz);
    infer_starts.resize(sz);
    infer_ends.resize(sz);
//    printf("sz %zu %f \n", trans_starts.size(), trans_starts[0]);
}



std::vector<int> cpu7_vector, cpu3_vector, cpu2_vector, cpu1_vector, cpu0_vector;
void syn_wait(syn_param& syn, int i){
//    pthread_mutex_lock(&syn.lock);
//    while (i > syn.num){
////        printf("%d, %d, %d %d\n", read_syn.num, create_syn.num, infer_syn.num, i);
////        pthread_cond_wait(&syn.cond, &syn.lock);
////        printf("lllock\n");
//        sleep(0);
//    }
//    pthread_mutex_unlock(&syn.lock);;
//    printf("lock\n");
    while(syn.f[i] != 4){
//        printf("%d\n", i);
        sleep(0);
    }
}
void syn_act(syn_param& syn, int i){
//    printf("act %d %d\n", syn.num, i);
//    pthread_mutex_lock(&syn.lock);
//    printf("act %d\n", syn.num, i);
//    if (i>=syn.num)
//        syn.num = i;
//    pthread_mutex_unlock(&syn.lock);
//    pthread_cond_signal(&syn.cond);

    syn.f[i] = 4;
}
int layer_next = 5;
pthread_mutex_t next_layer_lock = PTHREAD_MUTEX_INITIALIZER;
int select_next_layer(){
//    pthread_mutex_lock(&next_layer_lock);
    layer_next = layer_next + 1;
//    pthread_mutex_unlock(&next_layer_lock);
//    if(layer_next == 105||layer_next == 110||layer_next == 97)
//        layer_next ++;
    while (finish_set.count(layer_next))
    {
        layer_next++;
    }
    return layer_next;
}
int timeshow=0;
int finish_[]={};//{16,19,22};//{97, 105, 110};//{69,212,217};
void finish_set_init(){
    finish_set.clear();
    int len = cpu7_vector.size();
    for(int i=0; i<len; i++){
        finish_set.insert(cpu7_vector[i]);
    }

}
int layer_create_next = 5;
pthread_mutex_t next_layer_create_lock = PTHREAD_MUTEX_INITIALIZER;
int select_next_layer_create(){
    layer_create_next = layer_create_next + 1;
    return layer_create_next;
}

int param_finish = 0;
int param_finish_1 = 0;

double infer_start =0;
double infer_end =0;
double infer_time =0;
double dr_start =0;
double dr_end =0;
double dr_time =0;
double pipe_start =0;
double pipe_end =0;
double pipe_time =0;
double for_skp_time=0;
double for_cal_time=0;

void Delay_ms(int ms)
{
    struct timeval delay;
    delay.tv_sec = 0;
    delay.tv_usec = ms * 1000; // 20 ms
    select(0, NULL, NULL, NULL, &delay);
}

namespace ncnn {
int current_layer_idx_f2p =0;
int current_layer_idx_p2i=0;

class NetPrivate
{
public:
    NetPrivate(Option& _opt);

    Option& opt;

#if NCNN_VULKAN

    int upload_model();

#endif // NCNN_VULKAN

    friend class Extractor;
    int forward_layer(int layer_index, std::vector<Mat>& blob_mats, const Option& opt) const;

#if NCNN_VULKAN
    int forward_layer(int layer_index, std::vector<Mat>& blob_mats, std::vector<VkMat>& blob_mats_gpu, VkCompute& cmd, const Option& opt) const;
    int forward_layer(int layer_index, std::vector<Mat>& blob_mats, std::vector<VkMat>& blob_mats_gpu, std::vector<VkImageMat>& blob_mats_gpu_image, VkCompute& cmd, const Option& opt) const;
#endif // NCNN_VULKAN

    int convert_layout(Mat& bottom_blob, const Layer* layer, const Option& opt) const;

    int do_forward_layer(const Layer* layer, std::vector<Mat>& blob_mats, const Option& opt) const;
#if NCNN_VULKAN
    int do_forward_layer(const Layer* layer, std::vector<VkMat>& blob_mats_gpu, VkCompute& cmd, const Option& opt) const;
    int do_forward_layer(const Layer* layer, std::vector<VkImageMat>& blob_mats_gpu_image, VkCompute& cmd, const Option& opt) const;
#endif // NCNN_VULKAN

    void update_input_output_indexes();
#if NCNN_STRING
    void update_input_output_names();
#endif // NCNN_STRING

    std::vector<Blob> blobs;
    std::vector<Layer*> layers;

    std::vector<int> input_blob_indexes;
    std::vector<int> output_blob_indexes;
#if NCNN_STRING
    std::vector<const char*> input_blob_names;
    std::vector<const char*> output_blob_names;
#endif // NCNN_STRING

    std::vector<custom_layer_registry_entry> custom_layer_registry;

    PoolAllocator* local_blob_allocator;
    PoolAllocator* local_workspace_allocator;

#if NCNN_VULKAN
    const VulkanDevice* vkdev;

    VkAllocator* weight_vkallocator;
    VkAllocator* weight_staging_vkallocator;

    PipelineCache* pipeline_cache;
#endif // NCNN_VULKAN
};

NetPrivate::NetPrivate(Option& _opt)
    : opt(_opt)
{
    local_blob_allocator = 0;
    local_workspace_allocator = 0;

#if NCNN_VULKAN
    vkdev = 0;
    weight_vkallocator = 0;
    weight_staging_vkallocator = 0;
    pipeline_cache = 0;
#endif // NCNN_VULKAN
}

#if NCNN_VULKAN
int NetPrivate::upload_model()
{
    ncnn::VkTransfer cmd(vkdev);

    // create gpu device allocator if null
    if (!weight_vkallocator)
    {
        weight_vkallocator = new VkWeightAllocator(vkdev);
    }
    if (!weight_staging_vkallocator)
    {
        weight_staging_vkallocator = new VkWeightStagingAllocator(vkdev);
    }

    Option opt_upload = opt;
    opt_upload.blob_vkallocator = weight_vkallocator;
    opt_upload.workspace_vkallocator = weight_vkallocator;
    opt_upload.staging_vkallocator = weight_staging_vkallocator;

    for (size_t i = 0; i < layers.size(); i++)
    {
        if (layers[i]->support_vulkan)
        {
            int uret = layers[i]->upload_model(cmd, opt_upload);
            if (uret != 0)
            {
                NCNN_LOGE("layer upload_model %d failed", (int)i);
                return -1;
            }
        }
    }

    cmd.submit_and_wait();

    return 0;
}
#endif // NCNN_VULKAN

int NetPrivate::forward_layer(int layer_index, std::vector<Mat>& blob_mats, const Option& opt) const
{
//    std::vector<Mat> blob_mats1 = blob_mats;
//    const Layer* layer1 = layers[1];
//    int ret1 = do_forward_layer(layer1, blob_mats, opt);
//    const Layer* layer2 = layers[2];
//    int ret2 = do_forward_layer(layer2, blob_mats, opt);

    ///////////////////////////////////////////////////////////////////////////////////////////
    const Layer* layer = layers[layer_index];

    //     NCNN_LOGE("forward_layer %d %s", layer_index, layer->name.c_str());

    if (layer->one_blob_only)
    {
        // load bottom blob
        int bottom_blob_index = layer->bottoms[0];

        if (blob_mats[bottom_blob_index].dims == 0)
        {
            int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, opt);
            if (ret != 0)
                return ret;
        }
    }
    else
    {
        // load bottom blobs
        for (size_t i = 0; i < layer->bottoms.size(); i++)
        {
            int bottom_blob_index = layer->bottoms[i];

            if (blob_mats[bottom_blob_index].dims == 0)
            {
                int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, opt);
                if (ret != 0)
                    return ret;
            }
        }
    }

#if NCNN_BENCHMARK
    double start = get_current_time();
    Mat bottom_blob;
    if (layer->one_blob_only)
    {
        int bottom_blob_index = layer->bottoms[0];
        bottom_blob.dims = blob_mats[bottom_blob_index].dims;
        bottom_blob.w = blob_mats[bottom_blob_index].w;
        bottom_blob.h = blob_mats[bottom_blob_index].h;
        bottom_blob.c = blob_mats[bottom_blob_index].c;
        bottom_blob.elempack = blob_mats[bottom_blob_index].elempack;
        bottom_blob.elemsize = blob_mats[bottom_blob_index].elemsize;
    }
#endif
    double starts = ncnn::get_current_time();
    if(layer_index==1){
        infer_start = starts;
    }

//    printf("%d, %d\n", layer_index, ncnn::current_layer_idx_p2i);
//    pthread_mutex_lock(&infer_lock);
//    while (layer_index >= ncnn::current_layer_idx_p2i ){
//        pthread_cond_wait(&infer_cond, &infer_lock);
//    }
//    pthread_mutex_unlock(&infer_lock);
    syn_wait(infer_syn, layer_index);


    double ends = ncnn::get_current_time();
    double times = ends - starts;
//    if(times>0.0015)
//        printf("%d %f\n", layer_index, times);
    for_skp_time+=times;
//    fprintf(stderr, "             START do forward %s %d %d \n", layer->name.c_str(), layer_index, ncnn::current_layer_idx_p2i-1);

//    for(int i=0; i < blob_mats.size(); i++){
//        printf("%d, dims = %d %zu\n", i, blob_mats[0].dims, blob_mats.size());
//    }
//    printf("blob_mats.size()=%zu", blob_mats.size());
//    Delay_ms(10);
//    sleep(1);
    double start = ncnn::get_current_time();
//    printf("ffor\n");
//    std::cout<<"=====forward cpu[S]"<< layer_index<<"  "<<layer->name.c_str();
//    std::cout<<blob_mats.size()<<std::endl;
    int ret = do_forward_layer(layer, blob_mats, opt);
//    float* ptr = (float*)blob_mats[0].data;
//    printf("%d %f\n",layer_index,  *ptr);
//    printf("=====forward cpu%d %d %s\n", sched_getcpu(), layer_index, layer->name.c_str());
//    printf("=====forward cpu%d %d %s\n", sched_getcpu(), layer_index, layer->name.c_str());
//    std::cout<<"=====forward cpu"<< layer_index<<"  "<<layer->name.c_str()<<std::endl;
//        printf("=====forward %s\n", layer->name.c_str());
//    printf("for\n");
    double end = ncnn::get_current_time();


//    if(layer_index == 7)
//        printf("infer time = %f\n", end - start);

//    layers[layer_index]->destroy_pipeline(opt);
//    printf("before %lu\n", sizeof(* (layers[layer_index])));
//    layers[layer_index];
//    delete layer;
//    printf("after %lu\n", sizeof(*(layers[layer_index])));
//    printf("%s\n", layer->name.c_str());
//    fprintf(stderr, "%f\n", end - start);
//    fprintf(stderr, "%f\n", ends - starts);
//    fprintf(stderr, "%.2f %.2f\n", ends - starts,  end - start);
//    fprintf(stderr, "%s %f %f\n", layer->name.c_str(), ends - starts,  end - start);
//    printf("+=====+forward %d [cpu%d] %s %f \n",  layer_index, sched_getcpu(),layer->name.c_str(), end - start);
    if(save_time)
    {
        infer_starts[layer_index] = (start - save_start_time);
        infer_ends[layer_index] = (end - save_start_time);
    }

//    layer->destroy_pipeline(opt);
    double time = end - start;
//    if(layer->name == "conv2/3x3")
//        fprintf(stderr, "%6.3f,", time);
    for_cal_time+=time;
//    printf("   %f, %f\n", time, for_cal_time);
//    fprintf(stderr, "             do forward %7.2f %s %d %d\n", time, layer->name.c_str(), layer_index, ncnn::current_layer_idx_p2i-1);
//    fprintf(stderr, "fwrd: start %7.2f end: %7.2f %s %d %d time %f\n", start-dr_start, end-dr_start, layer->name.c_str(), layer_index, ncnn::current_layer_idx_p2i-1,time);

#if NCNN_BENCHMARK
    double end = get_current_time();
    if (layer->one_blob_only)
    {
        int top_blob_index = layer->tops[0];
        benchmark(layer, bottom_blob, blob_mats[top_blob_index], start, end);
    }
    else
    {
        benchmark(layer, start, end);
    }
#endif
    if (ret != 0)
        return ret;

    //     NCNN_LOGE("forward_layer %d %s done", layer_index, layer->name.c_str());
    //     const Mat& blob = blob_mats[layer->tops[0]];
    //     NCNN_LOGE("[%-2d %-16s %-16s]  %d    blobs count = %-3d   size = %-3d x %-3d", layer_index, layer->type.c_str(), layer->name.c_str(), layer->tops[0], blob.c, blob.h, blob.w);

    return 0;
}

#if NCNN_VULKAN
double tt=0;
int NetPrivate::forward_layer(int layer_index, std::vector<Mat>& blob_mats, std::vector<VkMat>& blob_mats_gpu, VkCompute& cmd, const Option& opt) const
{
    const Layer* layer = layers[layer_index];

    //     NCNN_LOGE("forward_layer %d %d %s", layer->support_vulkan, layer_index, layer->name.c_str());

    bool cmd_submit_and_wait = false;

    if (layer->one_blob_only)
    {
        // load bottom blob
        int bottom_blob_index = layer->bottoms[0];

        if (blob_mats_gpu[bottom_blob_index].dims == 0 && blob_mats[bottom_blob_index].dims == 0)
        {
            int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, blob_mats_gpu, cmd, opt);
            if (ret != 0)
                return ret;
        }

        if (layer->support_vulkan)
        {
//            double s = ncnn::get_current_time();
            if (blob_mats_gpu[bottom_blob_index].dims == 0)
            {
                // host to buffer
                cmd.record_upload(blob_mats[bottom_blob_index], blob_mats_gpu[bottom_blob_index], opt);

                if (opt.lightmode)
                {
                    // delete after taken in light mode
                    blob_mats[bottom_blob_index].release();
                }
            }
//            double e = ncnn::get_current_time();
//            printf("=+%d %f\n",blobs[bottom_blob_index].producer,  e-s);
//            tt+= e-s;
        }
        else
        {
            if (blob_mats[bottom_blob_index].dims == 0)
            {
                Option opt_download = opt;
                opt_download.use_packing_layout = layer->support_packing;

                // buffer to host
                cmd.record_download(blob_mats_gpu[bottom_blob_index], blob_mats[bottom_blob_index], opt_download);

                if (opt.lightmode)
                {
                    // delete after taken in light mode
                    blob_mats_gpu[bottom_blob_index].release();
                }

                cmd_submit_and_wait = true;
            }
        }
    }
    else
    {
        // load bottom blobs
        std::vector<VkMat> bottom_blobs(layer->bottoms.size());
        for (size_t i = 0; i < layer->bottoms.size(); i++)
        {
            int bottom_blob_index = layer->bottoms[i];

            if (blob_mats_gpu[bottom_blob_index].dims == 0 && blob_mats[bottom_blob_index].dims == 0)
            {
                int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, blob_mats_gpu, cmd, opt);
                if (ret != 0)
                    return ret;
            }

            if (layer->support_vulkan)
            {
//                double s = ncnn::get_current_time();
                if (blob_mats_gpu[bottom_blob_index].dims == 0)
                {
                    // host to buffer
                    cmd.record_upload(blob_mats[bottom_blob_index], blob_mats_gpu[bottom_blob_index], opt);

                    if (opt.lightmode)
                    {
                        // delete after taken in light mode
                        blob_mats[bottom_blob_index].release();
                    }
                }
//                double e = ncnn::get_current_time();
//                printf("=-%d %f\n",blobs[bottom_blob_index].producer, e-s);
//                tt+= e-s;
            }
            else
            {
                if (blob_mats[bottom_blob_index].dims == 0)
                {
                    Option opt_download = opt;
                    opt_download.use_packing_layout = layer->support_packing;

                    // buffer to host
                    cmd.record_download(blob_mats_gpu[bottom_blob_index], blob_mats[bottom_blob_index], opt_download);

                    if (opt.lightmode)
                    {
                        // delete after taken in light mode
                        blob_mats_gpu[bottom_blob_index].release();
                    }

                    cmd_submit_and_wait = true;
                }
            }
        }
    }

    if (cmd_submit_and_wait)
    {
        cmd.submit_and_wait();

#if NCNN_BENCHMARK
        std::vector<uint64_t> results(layer_index * 2);
        cmd.get_query_pool_results(0, layer_index * 2, results);
        for (int i = 0; i < layer_index; i++)
        {
            uint64_t start = results[i * 2];
            uint64_t end = results[i * 2 + 1];
            if (start == 0 || end == 0)
                continue;

            double duration_us = (end - start) * vkdev->info.timestamp_period() / 1000;
            NCNN_LOGE("%-24s %-30s %8.2lfus    |", layers[i]->type.c_str(), layers[i]->name.c_str(), duration_us);
        }
#endif // NCNN_BENCHMARK

        cmd.reset();
    }

    int ret;
    if (layer->support_vulkan)
    {
#if NCNN_BENCHMARK
        cmd.record_write_timestamp(layer_index * 2);
#endif
        double starts = ncnn::get_current_time();
        if(layer_index==1){
            infer_start = starts;
        }
        syn_wait(infer_syn, layer_index);
        double ends = ncnn::get_current_time();
        for_skp_time += ends-starts;

        double start = ncnn::get_current_time();
        ret = do_forward_layer(layer, blob_mats_gpu, cmd, opt);
        double end = ncnn::get_current_time();
        for_cal_time += end-start;
        tt+= end-start;
    //        printf("%d, %s, %f, %f\n", layer_index, layer->name.c_str(), end - start, end -strat0);
//        printf("%d, %f  %f\n", layer_index, end -start, tt);
#if NCNN_BENCHMARK
        cmd.record_write_timestamp(layer_index * 2 + 1);
#endif
    }
    else
    {
#if NCNN_BENCHMARK
        double start = get_current_time();
        Mat bottom_blob;
        if (layer->one_blob_only)
        {
            int bottom_blob_index = layer->bottoms[0];
            bottom_blob = blob_mats[bottom_blob_index].shape();
        }
#endif
        ret = do_forward_layer(layer, blob_mats, opt);
#if NCNN_BENCHMARK
        double end = get_current_time();
        if (layer->one_blob_only)
        {
            int top_blob_index = layer->tops[0];
            benchmark(layer, bottom_blob, blob_mats[top_blob_index], start, end);
        }
        else
        {
            benchmark(layer, start, end);
        }
#endif
    }
    if (ret != 0)
        return ret;

    //     NCNN_LOGE("forward_layer %d %d %s done", layer->support_vulkan, layer_index, layer->name.c_str());

    return 0;
}

int NetPrivate::forward_layer(int layer_index, std::vector<Mat>& blob_mats, std::vector<VkMat>& blob_mats_gpu, std::vector<VkImageMat>& blob_mats_gpu_image, VkCompute& cmd, const Option& opt) const
{
    const Layer* layer = layers[layer_index];

    //     NCNN_LOGE("forward_layer %d %d %s", layer->support_vulkan, layer_index, layer->name.c_str());

    bool cmd_submit_and_wait = false;
    bool image_allocation_failed = false;

IMAGE_ALLOCATION_FAILED:

    if (image_allocation_failed)
    {
#if NCNN_STRING
        NCNN_LOGE("forward_layer %d %s image allocation failed, fallback to cpu", layer_index, layer->name.c_str());
#else
        NCNN_LOGE("forward_layer %d image allocation failed, fallback to cpu", layer_index);
#endif
    }

    if (layer->one_blob_only)
    {
        // load bottom blob
        int bottom_blob_index = layer->bottoms[0];

        if (blob_mats_gpu_image[bottom_blob_index].dims == 0 && blob_mats_gpu[bottom_blob_index].dims == 0 && blob_mats[bottom_blob_index].dims == 0)
        {
            int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, blob_mats_gpu, blob_mats_gpu_image, cmd, opt);
            if (ret != 0)
                return ret;
        }

        if (layer->support_vulkan && !image_allocation_failed)
        {
            if (layer->support_image_storage)
            {
                if (blob_mats_gpu_image[bottom_blob_index].dims == 0)
                {
                    if (blob_mats_gpu[bottom_blob_index].dims == 0)
                    {
                        // host to image
                        cmd.record_upload(blob_mats[bottom_blob_index], blob_mats_gpu_image[bottom_blob_index], opt);

                        if (blob_mats_gpu_image[bottom_blob_index].empty())
                        {
                            image_allocation_failed = true;
                            goto IMAGE_ALLOCATION_FAILED;
                        }

                        if (opt.lightmode)
                        {
                            // delete after taken in light mode
                            blob_mats[bottom_blob_index].release();
                        }
                    }
                    else
                    {
                        // buffer to image
                        cmd.record_buffer_to_image(blob_mats_gpu[bottom_blob_index], blob_mats_gpu_image[bottom_blob_index], opt);

                        if (blob_mats_gpu_image[bottom_blob_index].empty())
                        {
                            image_allocation_failed = true;
                            goto IMAGE_ALLOCATION_FAILED;
                        }

                        if (opt.lightmode)
                        {
                            // delete after taken in light mode
                            blob_mats_gpu[bottom_blob_index].release();
                        }
                    }
                }
            }
            else
            {
                if (blob_mats_gpu[bottom_blob_index].dims == 0)
                {
                    if (blob_mats_gpu_image[bottom_blob_index].dims == 0)
                    {
                        // host to buffer
                        cmd.record_upload(blob_mats[bottom_blob_index], blob_mats_gpu[bottom_blob_index], opt);

                        if (opt.lightmode)
                        {
                            // delete after taken in light mode
                            blob_mats[bottom_blob_index].release();
                        }
                    }
                    else
                    {
                        // image to buffer
                        cmd.record_image_to_buffer(blob_mats_gpu_image[bottom_blob_index], blob_mats_gpu[bottom_blob_index], opt);

                        if (opt.lightmode)
                        {
                            // delete after taken in light mode
                            blob_mats_gpu_image[bottom_blob_index].release();
                        }
                    }
                }
            }
        }
        else
        {
            if (blob_mats[bottom_blob_index].dims == 0)
            {
                if (blob_mats_gpu_image[bottom_blob_index].dims == 0)
                {
                    // buffer to host
                    cmd.record_download(blob_mats_gpu[bottom_blob_index], blob_mats[bottom_blob_index], opt);

                    if (opt.lightmode)
                    {
                        // delete after taken in light mode
                        blob_mats_gpu[bottom_blob_index].release();
                    }

                    cmd_submit_and_wait = true;
                }
                else
                {
                    // image to host
                    cmd.record_download(blob_mats_gpu_image[bottom_blob_index], blob_mats[bottom_blob_index], opt);

                    if (opt.lightmode)
                    {
                        // delete after taken in light mode
                        blob_mats_gpu_image[bottom_blob_index].release();
                    }

                    cmd_submit_and_wait = true;
                }
            }
        }
    }
    else
    {
        // load bottom blobs
        std::vector<VkImageMat> bottom_blobs(layer->bottoms.size());
        for (size_t i = 0; i < layer->bottoms.size(); i++)
        {
            int bottom_blob_index = layer->bottoms[i];

            if (blob_mats_gpu_image[bottom_blob_index].dims == 0 && blob_mats_gpu[bottom_blob_index].dims == 0 && blob_mats[bottom_blob_index].dims == 0)
            {
                int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, blob_mats_gpu, blob_mats_gpu_image, cmd, opt);
                if (ret != 0)
                    return ret;
            }

            if (layer->support_vulkan && !image_allocation_failed)
            {
                if (layer->support_image_storage)
                {
                    if (blob_mats_gpu_image[bottom_blob_index].dims == 0)
                    {
                        if (blob_mats_gpu[bottom_blob_index].dims == 0)
                        {
                            // host to image
                            cmd.record_upload(blob_mats[bottom_blob_index], blob_mats_gpu_image[bottom_blob_index], opt);

                            if (blob_mats_gpu_image[bottom_blob_index].empty())
                            {
                                image_allocation_failed = true;
                                goto IMAGE_ALLOCATION_FAILED;
                            }

                            if (opt.lightmode)
                            {
                                // delete after taken in light mode
                                blob_mats[bottom_blob_index].release();
                            }
                        }
                        else
                        {
                            // buffer to image
                            cmd.record_buffer_to_image(blob_mats_gpu[bottom_blob_index], blob_mats_gpu_image[bottom_blob_index], opt);

                            if (blob_mats_gpu_image[bottom_blob_index].empty())
                            {
                                image_allocation_failed = true;
                                goto IMAGE_ALLOCATION_FAILED;
                            }

                            if (opt.lightmode)
                            {
                                // delete after taken in light mode
                                blob_mats_gpu[bottom_blob_index].release();
                            }
                        }
                    }
                }
                else
                {
                    if (blob_mats_gpu[bottom_blob_index].dims == 0)
                    {
                        if (blob_mats_gpu_image[bottom_blob_index].dims == 0)
                        {
                            // host to buffer
                            cmd.record_upload(blob_mats[bottom_blob_index], blob_mats_gpu[bottom_blob_index], opt);

                            if (opt.lightmode)
                            {
                                // delete after taken in light mode
                                blob_mats[bottom_blob_index].release();
                            }
                        }
                        else
                        {
                            // image to buffer
                            cmd.record_image_to_buffer(blob_mats_gpu_image[bottom_blob_index], blob_mats_gpu[bottom_blob_index], opt);

                            if (opt.lightmode)
                            {
                                // delete after taken in light mode
                                blob_mats_gpu_image[bottom_blob_index].release();
                            }
                        }
                    }
                }
            }
            else
            {
                if (blob_mats[bottom_blob_index].dims == 0)
                {
                    if (blob_mats_gpu_image[bottom_blob_index].dims == 0)
                    {
                        // buffer to host
                        cmd.record_download(blob_mats_gpu[bottom_blob_index], blob_mats[bottom_blob_index], opt);

                        if (opt.lightmode)
                        {
                            // delete after taken in light mode
                            blob_mats_gpu[bottom_blob_index].release();
                        }

                        cmd_submit_and_wait = true;
                    }
                    else
                    {
                        // image to host
                        cmd.record_download(blob_mats_gpu_image[bottom_blob_index], blob_mats[bottom_blob_index], opt);

                        if (opt.lightmode)
                        {
                            // delete after taken in light mode
                            blob_mats_gpu_image[bottom_blob_index].release();
                        }

                        cmd_submit_and_wait = true;
                    }
                }
            }
        }
    }

    if (cmd_submit_and_wait)
    {
        cmd.submit_and_wait();

#if NCNN_BENCHMARK
        std::vector<uint64_t> results(layer_index * 2);
        cmd.get_query_pool_results(0, layer_index * 2, results);
        for (int i = 0; i < layer_index; i++)
        {
            uint64_t start = results[i * 2];
            uint64_t end = results[i * 2 + 1];
            if (start == 0 || end == 0)
                continue;

            double duration_us = (end - start) * vkdev->info.timestamp_period() / 1000;
            NCNN_LOGE("%-24s %-30s %8.2lfus    |", layers[i]->type.c_str(), layers[i]->name.c_str(), duration_us);
        }
#endif // NCNN_BENCHMARK

        cmd.reset();
    }

    int ret;
    if (layer->support_vulkan && !image_allocation_failed)
    {
#if NCNN_BENCHMARK
        cmd.record_write_timestamp(layer_index * 2);
#endif
        if (layer->support_image_storage)
        {
            ret = do_forward_layer(layer, blob_mats_gpu_image, cmd, opt);
            if (ret == -100)
            {
                image_allocation_failed = true;
                goto IMAGE_ALLOCATION_FAILED;
            }
        }
        else
        {
            ret = do_forward_layer(layer, blob_mats_gpu, cmd, opt);
        }
#if NCNN_BENCHMARK
        cmd.record_write_timestamp(layer_index * 2 + 1);
#endif
    }
    else
    {
#if NCNN_BENCHMARK
        double start = get_current_time();
        Mat bottom_blob;
        if (layer->one_blob_only)
        {
            int bottom_blob_index = layer->bottoms[0];
            bottom_blob = blob_mats[bottom_blob_index].shape();
        }
#endif
        ret = do_forward_layer(layer, blob_mats, opt);
#if NCNN_BENCHMARK
        double end = get_current_time();
        if (layer->one_blob_only)
        {
            int top_blob_index = layer->tops[0];
            benchmark(layer, bottom_blob, blob_mats[top_blob_index], start, end);
        }
        else
        {
            benchmark(layer, start, end);
        }
#endif
    }
    if (ret != 0)
        return ret;

    //     NCNN_LOGE("forward_layer %d %d %s done", layer->support_vulkan, layer_index, layer->name.c_str());

    return 0;
}
#endif // NCNN_VULKAN

int NetPrivate::convert_layout(Mat& bottom_blob, const Layer* layer, const Option& opt) const
{
    // clang-format off
    // *INDENT-OFF*
#if NCNN_ARM82
    if (opt.use_fp16_storage && cpu_support_arm_asimdhp())
    {
        if (bottom_blob.elembits() == 32 && layer->support_fp16_storage)
        {
            Mat bottom_blob_fp16;
            cast_float32_to_float16(bottom_blob, bottom_blob_fp16, opt);
            bottom_blob = bottom_blob_fp16;
        }
        if (bottom_blob.elembits() == 16 && !layer->support_fp16_storage)
        {
            Mat bottom_blob_fp32;
            cast_float16_to_float32(bottom_blob, bottom_blob_fp32, opt);
            bottom_blob = bottom_blob_fp32;
        }
    }
    else
#endif // NCNN_ARM82
#if NCNN_RVV
    if (opt.use_fp16_storage && cpu_support_riscv_v() && cpu_support_riscv_zfh())
    {
        if (bottom_blob.elembits() == 32 && layer->support_fp16_storage)
        {
            Mat bottom_blob_fp16;
            cast_float32_to_float16(bottom_blob, bottom_blob_fp16, opt);
            bottom_blob = bottom_blob_fp16;
        }
        if (bottom_blob.elembits() == 16 && !layer->support_fp16_storage)
        {
            Mat bottom_blob_fp32;
            cast_float16_to_float32(bottom_blob, bottom_blob_fp32, opt);
            bottom_blob = bottom_blob_fp32;
        }
    }
    else
#endif // NCNN_RVV
    if (opt.use_bf16_storage)
    {
        if (bottom_blob.elembits() == 32 && layer->support_bf16_storage)
        {
            Mat bottom_blob_bf16;
            cast_float32_to_bfloat16(bottom_blob, bottom_blob_bf16, opt);
            bottom_blob = bottom_blob_bf16;
        }
        if (bottom_blob.elembits() == 16 && !layer->support_bf16_storage)
        {
            Mat bottom_blob_fp32;
            cast_bfloat16_to_float32(bottom_blob, bottom_blob_fp32, opt);
            bottom_blob = bottom_blob_fp32;
        }
    }
    // *INDENT-ON*
    // clang-format on

    if (opt.use_packing_layout)
    {
        // resolve dst_elempack
        int dims = bottom_blob.dims;
        int elemcount = 0;
        if (dims == 1) elemcount = bottom_blob.elempack * bottom_blob.w;
        if (dims == 2) elemcount = bottom_blob.elempack * bottom_blob.h;
        if (dims == 3) elemcount = bottom_blob.elempack * bottom_blob.c;

        int elembits = bottom_blob.elembits();

        int dst_elempack = 1;
        if (layer->support_packing)
        {
            if (elembits == 32)
            {
#if NCNN_AVX2
                if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx2())
                    dst_elempack = 8;
                else if (elemcount % 4 == 0)
                    dst_elempack = 4;
#elif NCNN_RVV
                const int packn = ncnn::cpu_riscv_vlenb() / 4;
                if (elemcount % packn == 0)
                    dst_elempack = packn;
#else
                if (elemcount % 4 == 0)
                    dst_elempack = 4;
#endif
            }
            if (elembits == 16)
            {
#if NCNN_ARM82
                if (elemcount % 8 == 0 && opt.use_fp16_storage && opt.use_fp16_arithmetic && layer->support_fp16_storage)
                    dst_elempack = 8;
                else if (elemcount % 4 == 0)
                    dst_elempack = 4;
#elif NCNN_RVV
                const int packn = ncnn::cpu_riscv_vlenb() / 2;
                if (elemcount % packn == 0)
                    dst_elempack = packn;
#else
                if (elemcount % 4 == 0)
                    dst_elempack = 4;
#endif
            }
            if (elembits == 8)
            {
#if NCNN_RVV
                const int packn = ncnn::cpu_riscv_vlenb() / 1;
                if (elemcount % packn == 0)
                    dst_elempack = packn;
#else
                if (elemcount % 8 == 0)
                    dst_elempack = 8;
#endif
            }
        }

        if (bottom_blob.elempack != dst_elempack)
        {
            Mat bottom_blob_packed;
            convert_packing(bottom_blob, bottom_blob_packed, dst_elempack, opt);
            bottom_blob = bottom_blob_packed;
        }
    }

    return 0;
}

int NetPrivate::do_forward_layer(const Layer* layer, std::vector<Mat>& blob_mats, const Option& opt) const
{
//    printf("layer->one_blob_only %d opt.lightmode %d  layer->support_inplace %d\n", layer->one_blob_only, opt.lightmode, layer->support_inplace);
    if (layer->one_blob_only)
    {
        int bottom_blob_index = layer->bottoms[0];
        int top_blob_index = layer->tops[0];

        Mat& bottom_blob_ref = blob_mats[bottom_blob_index];
        Mat bottom_blob;
//        int csize = bottom_blob_ref.w*bottom_blob_ref.h*bottom_blob_ref.c*bottom_blob_ref.elempack*sizeof(bottom_blob_ref.data)/1024;
//        printf("%d bottom_blob_ref.dims=%d w=%d, h=%d, c=%d, el=%d, %lu %d\n", bottom_blob_index,bottom_blob_ref.dims,bottom_blob_ref.w, bottom_blob_ref.h, bottom_blob_ref.c, bottom_blob_ref.elempack, sizeof(bottom_blob_ref.data), csize);

        if (opt.lightmode)
        {
            // deep copy for inplace forward if data is shared
            if (layer->support_inplace && *bottom_blob_ref.refcount != 1)
            {
                bottom_blob = bottom_blob_ref.clone();
            }
        }
        if (bottom_blob.dims == 0)
        {
            bottom_blob = bottom_blob_ref;
        }

        convert_layout(bottom_blob, layer, opt);

        // forward
        int flas = 1;
        if (opt.lightmode && layer->support_inplace)
        {
            Mat& bottom_top_blob = bottom_blob;

//            struct timeval tv;
//            gettimeofday(&tv, NULL);
//            double start = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;

            int ret = layer->forward_inplace(bottom_top_blob, opt);

//            struct timeval tv1;
//            gettimeofday(&tv1, NULL);
//            double end = tv1.tv_sec * 1000.0 + tv1.tv_usec / 1000.0;
//            printf("forward inplace %f\n",  end - start);

            int ccsize = bottom_top_blob.w*bottom_top_blob.h*bottom_top_blob.c*bottom_top_blob.elempack*sizeof(bottom_top_blob.data)/1024;
//            printf("%d %s bottom_top_blob.dims=%d w=%d, h=%d, c=%d, el=%d, %lu %d\n", bottom_blob_index,layer->name.c_str(),bottom_top_blob.dims,bottom_top_blob.w, bottom_top_blob.h, bottom_top_blob.c, bottom_top_blob.elempack, sizeof(bottom_top_blob.data), ccsize);
//            printf("%s\n", layer->name.c_str());
//            printf("%d*%d*%d*%d\n", bottom_top_blob.w, bottom_top_blob.h, bottom_top_blob.c, bottom_top_blob.elempack);
//            printf("\n");
//            printf("%d\n", ccsize+0);

            if (ret != 0)
                return ret;

            // store top blob
            blob_mats[top_blob_index] = bottom_top_blob;
        }
        else
        {
            Mat top_blob;
            int ccsize = bottom_blob.w*bottom_blob.h*bottom_blob.c*bottom_blob.elempack*sizeof(bottom_blob.data)/1024;
//            printf("%d %s bottom_blob.dims=%d w=%d, h=%d, c=%d, el=%d, %lu %d\n", bottom_blob_index,layer->name.c_str(),bottom_blob.dims,bottom_blob.w, bottom_blob.h, bottom_blob.c, bottom_blob.elempack, sizeof(bottom_blob.data), ccsize);

//            struct timeval tv;
//            gettimeofday(&tv, NULL);
//            double start = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;

            int ret = layer->forward(bottom_blob, top_blob, opt);

//            int tsize = top_blob.w*top_blob.h*top_blob.c*top_blob.elempack*sizeof(top_blob.data)/1024;
//            printf("%d top_blob.dims=%d w=%d, h=%d, c=%d, el=%d, %lu %d\n", bottom_blob_index,top_blob.dims,top_blob.w, top_blob.h, top_blob.c, top_blob.elempack, sizeof(top_blob.data), tsize);
//            printf("%s\n", layer->name.c_str());
//            printf("%d*%d*%d*%d\n", bottom_blob.w, bottom_blob.h, bottom_blob.c, bottom_blob.elempack);
//            printf("%d*%d*%d*%d\t", top_blob.w, top_blob.h, top_blob.c, top_blob.elempack);
//            printf("%d\n", ccsize+tsize);

//            struct timeval tv1;
//            gettimeofday(&tv1, NULL);
//            double end = tv1.tv_sec * 1000.0 + tv1.tv_usec / 1000.0;
//            printf("forward %s %f\n",  layer->name.c_str(), end - start);
//            printf("%f\n", end - start);

            if (ret != 0)
                return ret;

            // store top blob
            blob_mats[top_blob_index] = top_blob;
        }

        if (opt.lightmode)
        {
            // delete after taken in light mode
            blob_mats[bottom_blob_index].release();
        }
    }
    else
    {
        std::vector<Mat> bottom_blobs(layer->bottoms.size());
        for (size_t i = 0; i < layer->bottoms.size(); i++)
        {
            int bottom_blob_index = layer->bottoms[i];

            Mat& bottom_blob_ref = blob_mats[bottom_blob_index];
            bottom_blobs[i].release();

            if (opt.lightmode)
            {
                // deep copy for inplace forward if data is shared
                if (layer->support_inplace && *bottom_blob_ref.refcount != 1)
                {
                    bottom_blobs[i] = bottom_blob_ref.clone();
                }
            }
            if (bottom_blobs[i].dims == 0)
            {
                bottom_blobs[i] = bottom_blob_ref;
            }

            convert_layout(bottom_blobs[i], layer, opt);
        }

        // forward
        if (opt.lightmode && layer->support_inplace)
        {
            std::vector<Mat>& bottom_top_blobs = bottom_blobs;
            int ret = layer->forward_inplace(bottom_top_blobs, opt);
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i = 0; i < layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats[top_blob_index] = bottom_top_blobs[i];
            }
        }
        else
        {
            std::vector<Mat> top_blobs(layer->tops.size());
            int ret = layer->forward(bottom_blobs, top_blobs, opt);
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i = 0; i < layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats[top_blob_index] = top_blobs[i];
            }
        }

        for (size_t i = 0; i < layer->bottoms.size(); i++)
        {
            int bottom_blob_index = layer->bottoms[i];

            if (opt.lightmode)
            {
                // delete after taken in light mode
                blob_mats[bottom_blob_index].release();
            }
        }
    }

    return 0;
}

#if NCNN_VULKAN
int NetPrivate::do_forward_layer(const Layer* layer, std::vector<VkMat>& blob_mats_gpu, VkCompute& cmd, const Option& opt) const
{
    if (layer->one_blob_only)
    {
        // load bottom blob
        int bottom_blob_index = layer->bottoms[0];
        int top_blob_index = layer->tops[0];

        VkMat& bottom_blob_ref = blob_mats_gpu[bottom_blob_index];
        VkMat bottom_blob;

        if (opt.lightmode)
        {
            // deep copy for inplace forward if data is shared
            if (layer->support_inplace && *bottom_blob_ref.refcount != 1)
            {
                cmd.record_clone(bottom_blob_ref, bottom_blob, opt);
                //                     NCNN_LOGE("clone %p[+%lu] %p[+%lu]", bottom_blob_ref.buffer(), bottom_blob_ref.buffer_offset(), bottom_blob.buffer(), bottom_blob.buffer_offset());
            }
        }
        if (bottom_blob.dims == 0)
        {
            bottom_blob = bottom_blob_ref;
        }

        // forward
        if (opt.lightmode && layer->support_inplace)
        {
            VkMat& bottom_top_blob = bottom_blob;
            int ret = layer->forward_inplace(bottom_top_blob, cmd, opt);
            if (ret != 0)
                return ret;

            // store top blob
            blob_mats_gpu[top_blob_index] = bottom_top_blob;
        }
        else
        {
            VkMat top_blob;
            int ret = layer->forward(bottom_blob, top_blob, cmd, opt);
            if (ret != 0)
                return ret;

            // store top blob
            blob_mats_gpu[top_blob_index] = top_blob;
        }

        if (opt.lightmode)
        {
            // delete after taken in light mode
            blob_mats_gpu[bottom_blob_index].release();
        }
    }
    else
    {
        // load bottom blobs
        std::vector<VkMat> bottom_blobs(layer->bottoms.size());
        for (size_t i = 0; i < layer->bottoms.size(); i++)
        {
            int bottom_blob_index = layer->bottoms[i];

            VkMat& bottom_blob_ref = blob_mats_gpu[bottom_blob_index];
            bottom_blobs[i].release();

            if (opt.lightmode)
            {
                // deep copy for inplace forward if data is shared
                if (layer->support_inplace && *bottom_blob_ref.refcount != 1)
                {
                    cmd.record_clone(bottom_blob_ref, bottom_blobs[i], opt);
                    //                         NCNN_LOGE("clone %p[+%lu] %p[+%lu]", bottom_blob_ref.buffer(), bottom_blob_ref.buffer_offset(), bottom_blobs[i].buffer(), bottom_blobs[i].buffer_offset());
                }
            }
            if (bottom_blobs[i].dims == 0)
            {
                bottom_blobs[i] = bottom_blob_ref;
            }
        }

        // forward
        if (opt.lightmode && layer->support_inplace)
        {
            std::vector<VkMat>& bottom_top_blobs = bottom_blobs;
            int ret = layer->forward_inplace(bottom_top_blobs, cmd, opt);
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i = 0; i < layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats_gpu[top_blob_index] = bottom_top_blobs[i];
            }
        }
        else
        {
            std::vector<VkMat> top_blobs(layer->tops.size());

//            struct timeval tv0;
//            gettimeofday(&tv0, NULL);
//            double start = tv0.tv_sec * 1000.0 + tv0.tv_usec / 1000.0;

            int ret = layer->forward(bottom_blobs, top_blobs, cmd, opt);

            //            int tsize = top_blob.w*top_blob.h*top_blob.c*top_blob.elempack*sizeof(top_blob.data)/1024;
            //            printf("%d top_blob.dims=%d w=%d, h=%d, c=%d, el=%d, %lu %d\n", bottom_blob_index,top_blob.dims,top_blob.w, top_blob.h, top_blob.c, top_blob.elempack, sizeof(top_blob.data), tsize);
            //            printf("%s\n", layer->name.c_str());
            //            printf("%d*%d*%d*%d\n", bottom_blob.w, bottom_blob.h, bottom_blob.c, bottom_blob.elempack);
            //            printf("%d*%d*%d*%d\t", top_blob.w, top_blob.h, top_blob.c, top_blob.elempack);
            //            printf("%d\n", ccsize+tsize);

//            struct timeval tv1;
//            gettimeofday(&tv1, NULL);
//            double end = tv1.tv_sec * 1000.0 + tv1.tv_usec / 1000.0;
//            printf("forward %s %f\n",  layer->name.c_str(), end - start);
//            printf("%f\n", end - start);
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i = 0; i < layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats_gpu[top_blob_index] = top_blobs[i];
            }
        }

        for (size_t i = 0; i < layer->bottoms.size(); i++)
        {
            int bottom_blob_index = layer->bottoms[i];

            if (opt.lightmode)
            {
                // delete after taken in light mode
                blob_mats_gpu[bottom_blob_index].release();
            }
        }
    }

    return 0;
}

int NetPrivate::do_forward_layer(const Layer* layer, std::vector<VkImageMat>& blob_mats_gpu_image, VkCompute& cmd, const Option& opt) const
{
    if (layer->one_blob_only)
    {
        // load bottom blob
        int bottom_blob_index = layer->bottoms[0];
        int top_blob_index = layer->tops[0];

        VkImageMat& bottom_blob_ref = blob_mats_gpu_image[bottom_blob_index];
        VkImageMat bottom_blob;

        if (opt.lightmode)
        {
            // deep copy for inplace forward if data is shared
            if (layer->support_inplace && *bottom_blob_ref.refcount != 1)
            {
                cmd.record_clone(bottom_blob_ref, bottom_blob, opt);
                //                         NCNN_LOGE("clone %p[+%lu] %p[+%lu]", bottom_blob_ref.buffer(), bottom_blob_ref.buffer_offset(), bottom_blob.buffer(), bottom_blob.buffer_offset());
            }
        }
        if (bottom_blob.dims == 0)
        {
            bottom_blob = bottom_blob_ref;
        }

        // forward
        if (opt.lightmode && layer->support_inplace)
        {
            VkImageMat& bottom_top_blob = bottom_blob;
            int ret = layer->forward_inplace(bottom_top_blob, cmd, opt);
            if (ret != 0)
                return ret;

            // store top blob
            blob_mats_gpu_image[top_blob_index] = bottom_top_blob;
        }
        else
        {
            VkImageMat top_blob;
            int ret = layer->forward(bottom_blob, top_blob, cmd, opt);
            if (ret != 0)
                return ret;

            // store top blob
            blob_mats_gpu_image[top_blob_index] = top_blob;
        }

        if (opt.lightmode)
        {
            // delete after taken in light mode
            blob_mats_gpu_image[bottom_blob_index].release();
        }
    }
    else
    {
        // load bottom blobs
        std::vector<VkImageMat> bottom_blobs(layer->bottoms.size());
        for (size_t i = 0; i < layer->bottoms.size(); i++)
        {
            int bottom_blob_index = layer->bottoms[i];

            VkImageMat& bottom_blob_ref = blob_mats_gpu_image[bottom_blob_index];

            if (opt.lightmode)
            {
                // deep copy for inplace forward if data is shared
                if (layer->support_inplace && *bottom_blob_ref.refcount != 1)
                {
                    cmd.record_clone(bottom_blob_ref, bottom_blobs[i], opt);
                    //                             NCNN_LOGE("clone %p[+%lu] %p[+%lu]", bottom_blob_ref.buffer(), bottom_blob_ref.buffer_offset(), bottom_blobs[i].buffer(), bottom_blobs[i].buffer_offset());
                }
            }
            if (bottom_blobs[i].dims == 0)
            {
                bottom_blobs[i] = bottom_blob_ref;
            }
        }

        // forward
        if (opt.lightmode && layer->support_inplace)
        {
            std::vector<VkImageMat>& bottom_top_blobs = bottom_blobs;
            int ret = layer->forward_inplace(bottom_top_blobs, cmd, opt);
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i = 0; i < layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats_gpu_image[top_blob_index] = bottom_top_blobs[i];
            }
        }
        else
        {
            std::vector<VkImageMat> top_blobs(layer->tops.size());
            int ret = layer->forward(bottom_blobs, top_blobs, cmd, opt);
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i = 0; i < layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats_gpu_image[top_blob_index] = top_blobs[i];
            }
        }

        for (size_t i = 0; i < layer->bottoms.size(); i++)
        {
            int bottom_blob_index = layer->bottoms[i];

            if (opt.lightmode)
            {
                // delete after taken in light mode
                blob_mats_gpu_image[bottom_blob_index].release();
            }
        }
    }

    return 0;
}
#endif // NCNN_VULKAN

void NetPrivate::update_input_output_indexes()
{
    input_blob_indexes.clear();
    output_blob_indexes.clear();

    for (size_t i = 0; i < layers.size(); i++)
    {
        if (layers[i]->typeindex == LayerType::Input)
        {
            int blob_index = layers[i]->tops[0];
            input_blob_indexes.push_back(blob_index);
        }
    }

    for (size_t i = 0; i < blobs.size(); i++)
    {
        if (blobs[i].producer != -1 && blobs[i].consumer == -1)
        {
            output_blob_indexes.push_back(i);
        }
    }
}

#if NCNN_STRING
void NetPrivate::update_input_output_names()
{
    input_blob_names.clear();
    output_blob_names.clear();

    for (size_t i = 0; i < input_blob_indexes.size(); i++)
    {
        int blob_index = input_blob_indexes[i];
        input_blob_names.push_back(blobs[blob_index].name.c_str());
    }

    for (size_t i = 0; i < output_blob_indexes.size(); i++)
    {
        int blob_index = output_blob_indexes[i];
        output_blob_names.push_back(blobs[blob_index].name.c_str());
    }
}
#endif // NCNN_STRING

Net::Net()
    : d(new NetPrivate(opt))
{
}

Net::~Net()
{
    clear();

    delete d;
}

Net::Net(const Net&)
    : d(0)
{
}

Net& Net::operator=(const Net&)
{
    return *this;
}

#if NCNN_STRING
int Net::register_custom_layer(const char* type, layer_creator_func creator, layer_destroyer_func destroyer, void* userdata)
{
    int typeindex = layer_to_index(type);
    if (typeindex != -1)
    {
        NCNN_LOGE("can not register build-in layer type %s", type);
        return -1;
    }

    int custom_index = custom_layer_to_index(type);
    if (custom_index == -1)
    {
        struct custom_layer_registry_entry entry = {type, creator, destroyer, userdata};
        d->custom_layer_registry.push_back(entry);
    }
    else
    {
        NCNN_LOGE("overwrite existing custom layer type %s", type);
        d->custom_layer_registry[custom_index].name = type;
        d->custom_layer_registry[custom_index].creator = creator;
        d->custom_layer_registry[custom_index].destroyer = destroyer;
        d->custom_layer_registry[custom_index].userdata = userdata;
    }

    return 0;
}
#endif // NCNN_STRING

int Net::register_custom_layer(int index, layer_creator_func creator, layer_destroyer_func destroyer, void* userdata)
{
    int custom_index = index & ~LayerType::CustomBit;
    if (index == custom_index)
    {
        NCNN_LOGE("can not register build-in layer index %d", custom_index);
        return -1;
    }

    if ((int)d->custom_layer_registry.size() <= custom_index)
    {
#if NCNN_STRING
        struct custom_layer_registry_entry dummy = {"", 0, 0, 0};
#else
        struct custom_layer_registry_entry dummy = {0, 0, 0};
#endif // NCNN_STRING
        d->custom_layer_registry.resize(custom_index + 1, dummy);
    }

    if (d->custom_layer_registry[custom_index].creator)
    {
        NCNN_LOGE("overwrite existing custom layer index %d", custom_index);
    }

    d->custom_layer_registry[custom_index].creator = creator;
    d->custom_layer_registry[custom_index].destroyer = destroyer;
    d->custom_layer_registry[custom_index].userdata = userdata;
    return 0;
}

#if NCNN_STRING
int Net::load_param(const DataReader& dr)
{
#define SCAN_VALUE(fmt, v)                \
    if (dr.scan(fmt, &v) != 1)            \
    {                                     \
        NCNN_LOGE("parse " #v " failed"); \
        return -1;                        \
    }

    int magic = 0;
    SCAN_VALUE("%d", magic)
    if (magic != 7767517)
    {
        NCNN_LOGE("param is too old, please regenerate");
        return -1;
    }

    // parse
    int layer_count = 0;
    int blob_count = 0;
    SCAN_VALUE("%d", layer_count)
    SCAN_VALUE("%d", blob_count)
    if (layer_count <= 0 || blob_count <= 0)
    {
        NCNN_LOGE("invalid layer_count or blob_count");
        return -1;
    }

    d->layers.resize((size_t)layer_count);
    d->blobs.resize((size_t)blob_count);

#if NCNN_VULKAN
    // TODO enable gpu when bf16 conversion implemented
    if (opt.use_bf16_storage)
        opt.use_vulkan_compute = false;

    if (opt.use_vulkan_compute)
    {
        if (!d->vkdev) d->vkdev = get_gpu_device();
        if (!d->vkdev) opt.use_vulkan_compute = false; // no vulkan device, fallback to cpu
    }
    if (opt.use_vulkan_compute)
    {
        // sanitize use options
        if (!d->vkdev->info.support_fp16_packed()) opt.use_fp16_packed = false;
        if (!d->vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;
        if (!d->vkdev->info.support_fp16_arithmetic()) opt.use_fp16_arithmetic = false;
        if (!d->vkdev->info.support_int8_storage()) opt.use_int8_storage = false;
        if (!d->vkdev->info.support_int8_arithmetic()) opt.use_int8_arithmetic = false;

        if (d->vkdev->info.bug_buffer_image_load_zero()) opt.use_image_storage = false;

        // fp16a makes no sense when fp16 storage disabled
        if (!opt.use_fp16_packed && !opt.use_fp16_storage) opt.use_fp16_arithmetic = false;
    }
    else
    {
        // fp16a makes no sense when fp16 storage disabled
        if (!opt.use_fp16_storage) opt.use_fp16_arithmetic = false;
    }
#endif // NCNN_VULKAN

    ParamDict pd;

    int blob_index = 0;
    for (int i = 0; i < layer_count; i++)
    {
        char layer_type[256];
        char layer_name[256];
        int bottom_count = 0;
        int top_count = 0;
        SCAN_VALUE("%255s", layer_type)
        SCAN_VALUE("%255s", layer_name)
        SCAN_VALUE("%d", bottom_count)
        SCAN_VALUE("%d", top_count)

        Layer* layer = create_layer(layer_type);
        if (!layer)
        {
            layer = create_custom_layer(layer_type);
        }
        if (!layer)
        {
            NCNN_LOGE("layer %s not exists or registered", layer_type);
            clear();
            return -1;
        }

#if NCNN_VULKAN
        if (opt.use_vulkan_compute)
            layer->vkdev = d->vkdev;
#endif // NCNN_VULKAN

        layer->type = std::string(layer_type);
        layer->name = std::string(layer_name);
        //         NCNN_LOGE("new layer %d %s", i, layer_name);

        layer->bottoms.resize(bottom_count);

        for (int j = 0; j < bottom_count; j++)
        {
            char bottom_name[256];
            SCAN_VALUE("%255s", bottom_name)

            int bottom_blob_index = find_blob_index_by_name(bottom_name);
            if (bottom_blob_index == -1)
            {
                Blob& blob = d->blobs[blob_index];

                bottom_blob_index = blob_index;

                blob.name = std::string(bottom_name);
                //                 NCNN_LOGE("new blob %s", bottom_name);

                blob_index++;
            }

            Blob& blob = d->blobs[bottom_blob_index];

            blob.consumer = i;

            layer->bottoms[j] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int j = 0; j < top_count; j++)
        {
            Blob& blob = d->blobs[blob_index];

            char blob_name[256];
            SCAN_VALUE("%255s", blob_name)

            blob.name = std::string(blob_name);
            //             NCNN_LOGE("new blob %s", blob_name);

            blob.producer = i;

            layer->tops[j] = blob_index;

            blob_index++;
        }

        // layer specific params
        int pdlr = pd.load_param(dr);
        if (pdlr != 0)
        {
            NCNN_LOGE("ParamDict load_param %d %s failed", i, layer->name.c_str());
            continue;
        }

        if (layer->support_int8_storage)
        {
            // no int8 gpu support yet
            opt.use_vulkan_compute = false;
        }

        // pull out top shape hints
        Mat shape_hints = pd.get(30, Mat());
        if (!shape_hints.empty())
        {
            const int* psh = shape_hints;
            for (int j = 0; j < top_count; j++)
            {
                Blob& blob = d->blobs[layer->tops[j]];

                int dims = psh[0];
                if (dims == 1)
                {
                    blob.shape = Mat(psh[1], (void*)0, 4u, 1);
                }
                if (dims == 2)
                {
                    blob.shape = Mat(psh[1], psh[2], (void*)0, 4u, 1);
                }
                if (dims == 3)
                {
                    blob.shape = Mat(psh[1], psh[2], psh[3], (void*)0, 4u, 1);
                }

                psh += 4;
            }
        }

        // set bottom and top shape hints
        layer->bottom_shapes.resize(bottom_count);
        for (int j = 0; j < bottom_count; j++)
        {
            layer->bottom_shapes[j] = d->blobs[layer->bottoms[j]].shape;
        }

        layer->top_shapes.resize(top_count);
        for (int j = 0; j < top_count; j++)
        {
            layer->top_shapes[j] = d->blobs[layer->tops[j]].shape;
        }

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
            NCNN_LOGE("layer load_param %d %s failed", i, layer->name.c_str());
            continue;
        }

        d->layers[i] = layer;
    }

    d->update_input_output_indexes();
    d->update_input_output_names();

#undef SCAN_VALUE
    return 0;
}
#endif // NCNN_STRING

int Net::load_param_bin(const DataReader& dr)
{
#define READ_VALUE(buf)                            \
    if (dr.read(&buf, sizeof(buf)) != sizeof(buf)) \
    {                                              \
        NCNN_LOGE("read " #buf " failed");         \
        return -1;                                 \
    }

    int magic = 0;
    READ_VALUE(magic)
    if (magic != 7767517)
    {
        NCNN_LOGE("param is too old, please regenerate");
        return -1;
    }

    int layer_count = 0;
    int blob_count = 0;
    READ_VALUE(layer_count)
    READ_VALUE(blob_count)
    if (layer_count <= 0 || blob_count <= 0)
    {
        NCNN_LOGE("invalid layer_count or blob_count");
        return -1;
    }

    d->layers.resize(layer_count);
    d->blobs.resize(blob_count);

#if NCNN_VULKAN
    // TODO enable gpu when bf16 conversion implemented
    if (opt.use_bf16_storage)
        opt.use_vulkan_compute = false;

    if (opt.use_vulkan_compute)
    {
        if (!d->vkdev) d->vkdev = get_gpu_device();
        if (!d->vkdev) opt.use_vulkan_compute = false; // no vulkan device, fallback to cpu
    }
    if (opt.use_vulkan_compute)
    {
        // sanitize use options
        if (!d->vkdev->info.support_fp16_packed()) opt.use_fp16_packed = false;
        if (!d->vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;
        if (!d->vkdev->info.support_fp16_arithmetic()) opt.use_fp16_arithmetic = false;
        if (!d->vkdev->info.support_int8_storage()) opt.use_int8_storage = false;
        if (!d->vkdev->info.support_int8_arithmetic()) opt.use_int8_arithmetic = false;

        if (d->vkdev->info.bug_buffer_image_load_zero()) opt.use_image_storage = false;

        // fp16a makes no sense when fp16 storage disabled
        if (!opt.use_fp16_packed && !opt.use_fp16_storage) opt.use_fp16_arithmetic = false;
    }
    else
    {
        // fp16a makes no sense when fp16 storage disabled
        if (!opt.use_fp16_storage) opt.use_fp16_arithmetic = false;
    }
#endif // NCNN_VULKAN

    ParamDict pd;

    for (int i = 0; i < layer_count; i++)
    {
        int typeindex;
        int bottom_count;
        int top_count;
        READ_VALUE(typeindex)
        READ_VALUE(bottom_count)
        READ_VALUE(top_count)

        Layer* layer = create_layer(typeindex);
        if (!layer)
        {
            int custom_index = typeindex & ~LayerType::CustomBit;
            layer = create_custom_layer(custom_index);
        }
        if (!layer)
        {
            NCNN_LOGE("layer %d not exists or registered", typeindex);
            clear();
            return -1;
        }

#if NCNN_VULKAN
        if (opt.use_vulkan_compute)
            layer->vkdev = d->vkdev;
#endif // NCNN_VULKAN

        //         layer->type = std::string(layer_type);
        //         layer->name = std::string(layer_name);
        //         NCNN_LOGE("new layer %d", typeindex);

        layer->bottoms.resize(bottom_count);
        for (int j = 0; j < bottom_count; j++)
        {
            int bottom_blob_index;
            READ_VALUE(bottom_blob_index)

            Blob& blob = d->blobs[bottom_blob_index];

            blob.consumer = i;

            layer->bottoms[j] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int j = 0; j < top_count; j++)
        {
            int top_blob_index;
            READ_VALUE(top_blob_index)

            Blob& blob = d->blobs[top_blob_index];

            //             blob.name = std::string(blob_name);
            //             NCNN_LOGE("new blob %s", blob_name);

            blob.producer = i;

            layer->tops[j] = top_blob_index;
        }

        // layer specific params
        int pdlr = pd.load_param_bin(dr);
        if (pdlr != 0)
        {
#if NCNN_STRING
            NCNN_LOGE("ParamDict load_param %d %s failed", i, layer->name.c_str());
#else
            NCNN_LOGE("ParamDict load_param %d failed", i);
#endif
            continue;
        }

        if (layer->support_int8_storage)
        {
            // no int8 gpu support yet
            opt.use_vulkan_compute = false;
        }

        // pull out top blob shape hints
        Mat shape_hints = pd.get(30, Mat());
        if (!shape_hints.empty())
        {
            const int* psh = shape_hints;
            for (int j = 0; j < top_count; j++)
            {
                Blob& blob = d->blobs[layer->tops[j]];

                int dims = psh[0];
                if (dims == 1)
                {
                    blob.shape = Mat(psh[1], (void*)0, 4u, 1);
                }
                if (dims == 2)
                {
                    blob.shape = Mat(psh[1], psh[2], (void*)0, 4u, 1);
                }
                if (dims == 3)
                {
                    blob.shape = Mat(psh[1], psh[2], psh[3], (void*)0, 4u, 1);
                }

                psh += 4;
            }
        }

        // set bottom and top shape hints
        layer->bottom_shapes.resize(bottom_count);
        for (int j = 0; j < bottom_count; j++)
        {
            layer->bottom_shapes[j] = d->blobs[layer->bottoms[j]].shape;
        }

        layer->top_shapes.resize(top_count);
        for (int j = 0; j < top_count; j++)
        {
            layer->top_shapes[j] = d->blobs[layer->tops[j]].shape;
        }

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
#if NCNN_STRING
            NCNN_LOGE("layer load_param %d %s failed", i, layer->name.c_str());
#else
            NCNN_LOGE("layer load_param %d failed", i);
#endif
            continue;
        }

        d->layers[i] = layer;
    }

    d->update_input_output_indexes();

#undef READ_VALUE
    return 0;
}

int Net::load_model(const DataReader& dr)
{
    if (d->layers.empty())
    {
        NCNN_LOGE("network graph not ready");
        return -1;
    }

    int layer_count = (int)d->layers.size();

    // load file
    int ret = 0;

    double start_m = ncnn::get_current_time();
    ModelBinFromDataReader mb(dr);
    for (int i = 0; i < layer_count; i++)
    {
        Layer* layer = d->layers[i];

        //Here we found inconsistent content in the parameter file.
        if (!layer)
        {
            NCNN_LOGE("load_model error at layer %d, parameter file has inconsistent content.", i);
            ret = -1;
            break;
        }

        double start1 = ncnn::get_current_time();
        int lret = layer->load_model(mb);
        double end1 = ncnn::get_current_time();
        double time1 = end1 - start1;
//        fprintf(stderr, "             %d in load model time %f\n", i, time1);
        if (lret != 0)
        {
#if NCNN_STRING
            NCNN_LOGE("layer load_model %d %s failed", i, layer->name.c_str());
#else
            NCNN_LOGE("layer load_model %d failed", i);
#endif
            ret = -1;
            break;
        }

        if (layer->support_int8_storage)
        {
            // no int8 gpu support yet
            opt.use_vulkan_compute = false;
        }
    }

    double end_m = ncnn::get_current_time();
    double time_m = end_m - start_m;
    fprintf(stderr, "             load model time %f\n", time_m);
#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        if (!opt.pipeline_cache)
        {
            if (!d->pipeline_cache)
                d->pipeline_cache = new PipelineCache(d->vkdev);
            opt.pipeline_cache = d->pipeline_cache;
        }
    }
#endif // NCNN_VULKAN

    double start_p = ncnn::get_current_time();
    for (int i = 0; i < layer_count; i++)
    {
        Layer* layer = d->layers[i];

        Option opt1 = opt;
#if NCNN_VULKAN
        if (opt.use_vulkan_compute)
        {
            if (!layer->support_image_storage) opt1.use_image_storage = false;
        }
#endif // NCNN_VULKAN

        double start1 = ncnn::get_current_time();
        int cret = layer->create_pipeline(opt1);
        double end1 = ncnn::get_current_time();
        double time1 = end1 - start1;
        if (cret != 0)
        {
#if NCNN_STRING
            NCNN_LOGE("layer create_pipeline %d %s failed", i, layer->name.c_str());
#else
            NCNN_LOGE("layer create_pipeline %d failed", i);
#endif
            ret = -1;
            break;
        }
    }

    double end_p = ncnn::get_current_time();
    double time_p = end_p - start_p;
    fprintf(stderr, "             pipeline time %f\n", time_p);
    if (opt.use_local_pool_allocator)
    {
        if (opt.blob_allocator == 0)
        {
            if (!d->local_blob_allocator)
            {
                d->local_blob_allocator = new PoolAllocator;
                d->local_blob_allocator->set_size_compare_ratio(0.f);
            }
        }
        if (opt.workspace_allocator == 0)
        {
            if (!d->local_workspace_allocator)
            {
                d->local_workspace_allocator = new PoolAllocator;
                d->local_workspace_allocator->set_size_compare_ratio(0.5f);
            }
        }
    }

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        d->upload_model();
    }
#endif // NCNN_VULKAN
    return ret;
}

int Net::load_model_dr(const DataReader& dr)
{
    printf("_____    load_model_dr   tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());
    if (d->layers.empty())
    {
        NCNN_LOGE("network graph not ready");
        return -1;
    }

    int layer_count = (int)d->layers.size();

    // load file
    int ret = 0;

    double start_m = ncnn::get_current_time();
    ModelBinFromDataReader mb(dr);
    for (int i = 0; i < layer_count; i++)
    {
        Layer* layer = d->layers[i];
        Option opt1 = opt;

        //Here we found inconsistent content in the parameter file.
        if (!layer)
        {
            NCNN_LOGE("load_model error at layer %d, parameter file has inconsistent content.", i);
            ret = -1;
            break;
        }

//        fprintf(stderr, "            START %s %d in load model time \n", d->layers[i]->name.c_str(), ncnn::current_layer_idx_f2p-1);
//        printf("\n----------------start read %s\n", layer->name.c_str());
//        printf("%d start  %zu\n", i,DataReaderFromStdio_size);
//        if(i >0)
//        {
//            int tmp[] = {1,5,7,12,14,16,18,20,23,27,29,31,33,35,38,43,45,47,49,51,54,58,60,62,64,66,69,73,75,77,79,81,84,88,90,92,94,96,99,103,105,107,109,111,114,119,121,123,125,127,130,134,136,138,140,142,145};
//            int flag = 0;
//            for (int ii :tmp)
//            {
//                if(i == ii){
//                    flag = 1;
//                }
//                else
//                    flag = 0;
//            }
//            if (flag)
////                if (arm_weight_file_seek_Vectors[i - 1] < arm_weight_file_seek_Vectors[i])
//            {
//                load_weight_flag = 0;
//            }
//            else if(i == layer_count - 1){
//                load_weight_flag = 0;
//            }
//            else
//            {
//                load_weight_flag = 1;
//            }
//        }
        if(TEST_)
        {
//            printf("%zu, ", DataReaderFromStdio_size);
            DR_file_Vectors.push_back(DataReaderFromStdio_size);
        }
        double start1 = ncnn::get_current_time();
        int lret = layer->load_model(mb);
//        printf("%d %zu\n", i,DataReaderFromStdio_size);
//        syn_act(create_syn, i);
        double end1 = ncnn::get_current_time();
        double time1 = end1 - start1;

//        if(i == 7)
//            printf("load time = %f\n", end1 - start1);

//        pthread_mutex_lock(&pipe_lock);
//        ncnn::current_layer_idx_f2p =i +1;
//        pthread_mutex_unlock(&pipe_lock);
//        pthread_cond_signal(&pipe_cond);
        syn_act(create_syn, i);

//        fprintf(stderr, "%f\n", time1);
//        printf("----------------end read %s\n", layer->name.c_str());
//        fprintf(stderr, "load: start %7.2f end: %7.2f %s %d time %f\n", start1-dr_start, end1-dr_start, d->layers[i]->name.c_str(), ncnn::current_layer_idx_f2p-1,time1);

        //        fprintf(stderr, "             %s %d in load model time %f\n", d->layers[i]->name.c_str(), ncnn::current_layer_idx_f2p-1, time1);
        if (lret != 0)
        {
#if NCNN_STRING
            NCNN_LOGE("layer load_model %d %s failed", i, layer->name.c_str());
#else
            NCNN_LOGE("layer load_model %d failed", i);
#endif
            ret = -1;
            break;
        }

        if (layer->support_int8_storage)
        {
            // no int8 gpu support yet
            opt.use_vulkan_compute = false;
        }
    }

    double end_m = ncnn::get_current_time();
    double time_m = end_m - start_m;
//    fprintf(stderr, "load model time %f\n", time_m);
    return ret;
}
int Net::load_model_dr(FILE* fp)
{
    DataReaderFromStdio dr(fp);
    return load_model_dr(dr);
}
int Net::load_model_dr(const char* modelpath)
{
//    printf("load file %s\n", modelpath);
    FILE* fp = fopen(modelpath, "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen %s failed", modelpath);
        return -1;
    }

    int ret = load_model_dr(fp);
    fclose(fp);
    return ret;
}

int Net::load_model_dr_cpu0(const DataReader& dr)
{
    printf("_____    load_model_dr_cpu0   tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());
    if (d->layers.empty())
    {
        NCNN_LOGE("network graph not ready");
        return -1;
    }

    int layer_count = (int)d->layers.size();

    // load file
    int ret = 0;

    double start_m = ncnn::get_current_time();
    ModelBinFromDataReader mb(dr);
    for (int i = 0; i < layer_count; i++)
    {
        if(i%2==0){
            continue;
        }
        Layer* layer = d->layers[i];
        Option opt1 = opt;

        //Here we found inconsistent content in the parameter file.
        if (!layer)
        {
            NCNN_LOGE("load_model error at layer %d, parameter file has inconsistent content.", i);
            ret = -1;
            break;
        }

        //        fprintf(stderr, "            START %s %d in load model time \n", d->layers[i]->name.c_str(), ncnn::current_layer_idx_f2p-1);
        double start1 = ncnn::get_current_time();
        int lret = layer->load_model(mb);

        double end1 = ncnn::get_current_time();
        double time1 = end1 - start1;

        pthread_mutex_lock(&pipe_lock);
        ncnn::current_layer_idx_f2p =i +1;
        pthread_mutex_unlock(&pipe_lock);
        pthread_cond_signal(&pipe_cond);

        //        fprintf(stderr, "%f\n", time1);
        //        printf("%s\n", layer->name.c_str());
        //        fprintf(stderr, "load: start %7.2f end: %7.2f %s %d time %f\n", start1-dr_start, end1-dr_start, d->layers[i]->name.c_str(), ncnn::current_layer_idx_f2p-1,time1);

        //        fprintf(stderr, "             %s %d in load model time %f\n", d->layers[i]->name.c_str(), ncnn::current_layer_idx_f2p-1, time1);
        if (lret != 0)
        {
#if NCNN_STRING
            NCNN_LOGE("layer load_model %d %s failed", i, layer->name.c_str());
#else
            NCNN_LOGE("layer load_model %d failed", i);
#endif
            ret = -1;
            break;
        }

        if (layer->support_int8_storage)
        {
            // no int8 gpu support yet
            opt.use_vulkan_compute = false;
        }
    }

    double end_m = ncnn::get_current_time();
    double time_m = end_m - start_m;
    //    fprintf(stderr, "load model time %f\n", time_m);
    return ret;
}
int Net::load_model_dr_cpu0(FILE* fp)
{
    DataReaderFromStdio dr(fp);
    return load_model_dr(dr);
}
int Net::load_model_dr_cpu0(const char* modelpath)
{
    printf("load file %s\n", modelpath);
    FILE* fp = fopen(modelpath, "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen %s failed", modelpath);
        return -1;
    }

    int ret = load_model_dr(fp);
    fclose(fp);
    return ret;
}


int Net::load_model_dr_cpu1(const DataReader& dr)
{
    printf("_____    load_model_dr_cpu1   tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());
    if (d->layers.empty())
    {
        NCNN_LOGE("network graph not ready");
        return -1;
    }

    int layer_count = (int)d->layers.size();

    // load file
    int ret = 0;

    double start_m = ncnn::get_current_time();
    ModelBinFromDataReader mb(dr);
    for (int i = 0; i < layer_count; i++)
    {
        if(i%2!=0){
            continue;
        }
        Layer* layer = d->layers[i];
        Option opt1 = opt;

        //Here we found inconsistent content in the parameter file.
        if (!layer)
        {
            NCNN_LOGE("load_model error at layer %d, parameter file has inconsistent content.", i);
            ret = -1;
            break;
        }

        //        fprintf(stderr, "            START %s %d in load model time \n", d->layers[i]->name.c_str(), ncnn::current_layer_idx_f2p-1);
        double start1 = ncnn::get_current_time();
        int lret = layer->load_model(mb);

        double end1 = ncnn::get_current_time();
        double time1 = end1 - start1;

//        pthread_mutex_lock(&pipe_lock);
//        ncnn::current_layer_idx_f2p =i +1;
//        pthread_mutex_unlock(&pipe_lock);
//        pthread_cond_signal(&pipe_cond);

        //        fprintf(stderr, "%f\n", time1);
        //        printf("%s\n", layer->name.c_str());
        //        fprintf(stderr, "load: start %7.2f end: %7.2f %s %d time %f\n", start1-dr_start, end1-dr_start, d->layers[i]->name.c_str(), ncnn::current_layer_idx_f2p-1,time1);

        //        fprintf(stderr, "             %s %d in load model time %f\n", d->layers[i]->name.c_str(), ncnn::current_layer_idx_f2p-1, time1);
        if (lret != 0)
        {
#if NCNN_STRING
            NCNN_LOGE("layer load_model %d %s failed", i, layer->name.c_str());
#else
            NCNN_LOGE("layer load_model %d failed", i);
#endif
            ret = -1;
            break;
        }

        if (layer->support_int8_storage)
        {
            // no int8 gpu support yet
            opt.use_vulkan_compute = false;
        }
    }

    double end_m = ncnn::get_current_time();
    double time_m = end_m - start_m;
    //    fprintf(stderr, "load model time %f\n", time_m);
    return ret;
}
int Net::load_model_dr_cpu1(FILE* fp)
{
    DataReaderFromStdio dr(fp);
    return load_model_dr(dr);
}
int Net::load_model_dr_cpu1(const char* modelpath)
{
    printf("load file %s\n", modelpath);
    FILE* fp = fopen(modelpath, "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen %s failed", modelpath);
        return -1;
    }

    int ret = load_model_dr(fp);
    fclose(fp);
    return ret;
}

int Net::load_model_layer(const ModelBinFromDataReader& mb, int layer_idx){

    double start = get_current_time();
    // load file
    int ret = 0;
//    printf("=====read cpu%d %d %s\n", sched_getcpu(), layer_idx, d->layers[layer_idx]->name.c_str());

    Layer* layer = d->layers[layer_idx];
    Option opt1 = opt;

    //Here we found inconsistent content in the parameter file.
    if (!layer)
    {
        NCNN_LOGE("load_model error at layer %d, parameter file has inconsistent content.", layer_idx);
        ret = -1;
    }
    if(TEST_)
    {
        //        printf("%zu, ", DataReaderFromStdio_size);
        DR_file_Vectors.push_back(DataReaderFromStdio_size);
    }
    double start1 = ncnn::get_current_time();
//    printf("=====read cpu%d %d %s\n", sched_getcpu(), layer_idx, layer->name.c_str());

    int lret = layer->load_model(mb);

    double end1 = ncnn::get_current_time();
//    double time1 = end1 - start1;
//    printf("load model layer %d %f\n", layer_idx, time1);

//    printf("+=====+read %d [cpu%d] %s %f \n",  layer_idx, sched_getcpu(), layer->name.c_str(), end1 - start1);
    if(save_time)
    {
        read_starts[layer_idx] = (start1 - save_start_time);
        read_ends[layer_idx] = (end1 - save_start_time);
    }
    if (lret != 0)
    {
#if NCNN_STRING
        NCNN_LOGE("layer load_model %d %s failed", layer_idx, layer->name.c_str());
#else
        NCNN_LOGE("layer load_model %d failed", layer_idx);
#endif
        ret = -1;
    }

    if (layer->support_int8_storage)
    {
        // no int8 gpu support yet
        opt.use_vulkan_compute = false;
    }

    double end = get_current_time();
//    printf("r,%d,%d,%.3f\n", sched_getcpu(), layer_idx, end-start);
    return ret;
}

int Net::load_model_dr_layer(const DataReader& dr, int layer_idx)
{
    printf("_____    load_model_dr_layer   tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());
    if (d->layers.empty())
    {
        NCNN_LOGE("network graph not ready");
        return -1;
    }

    int layer_count = (int)d->layers.size();

    // load file
    int ret = 0;

    ModelBinFromDataReader mb(dr);

    Layer* layer = d->layers[layer_idx];
    Option opt1 = opt;

    //Here we found inconsistent content in the parameter file.
    if (!layer)
    {
        NCNN_LOGE("load_model error at layer %d, parameter file has inconsistent content.", layer_idx);
        ret = -1;
    }

    //        fprintf(stderr, "            START %s %d in load model time \n", d->layers[i]->name.c_str(), ncnn::current_layer_idx_f2p-1);
    double start1 = ncnn::get_current_time();
    int lret = layer->load_model(mb);

    double end1 = ncnn::get_current_time();
    double time1 = end1 - start1;
    printf("load model layer %d %f\n", layer_idx, time1);

    //        fprintf(stderr, "%f\n", time1);
    //        printf("%s\n", layer->name.c_str());
    //        fprintf(stderr, "load: start %7.2f end: %7.2f %s %d time %f\n", start1-dr_start, end1-dr_start, d->layers[i]->name.c_str(), ncnn::current_layer_idx_f2p-1,time1);

    //        fprintf(stderr, "             %s %d in load model time %f\n", d->layers[i]->name.c_str(), ncnn::current_layer_idx_f2p-1, time1);
    if (lret != 0)
    {
#if NCNN_STRING
        NCNN_LOGE("layer load_model %d %s failed", layer_idx, layer->name.c_str());
#else
        NCNN_LOGE("layer load_model %d failed", layer_idx);
#endif
        ret = -1;
    }

    if (layer->support_int8_storage)
    {
        // no int8 gpu support yet
        opt.use_vulkan_compute = false;
    }
    return ret;
}
int Net::load_model_dr_layer(FILE* fp, int layer_idx)
{
    DataReaderFromStdio dr(fp);
    return load_model_dr_layer(dr, layer_idx);
}
int Net::load_model_dr_layer(const char* modelpath, int layer_idx)
{
    printf("load file %s\n", modelpath);
    FILE* fp = fopen(modelpath, "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen %s failed", modelpath);
        return -1;
    }

    int ret = load_model_dr_layer(fp, layer_idx);
    fclose(fp);
    return ret;
}

int Net::load_model_pipe()
{
    printf("_____    load_model_pipe tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());
    int ret = 0;
    int layer_count = (int)d->layers.size();
#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        if (!opt.pipeline_cache)
        {
            if (!d->pipeline_cache)
                d->pipeline_cache = new PipelineCache(d->vkdev);
            opt.pipeline_cache = d->pipeline_cache;
        }
    }
#endif // NCNN_VULKAN

//    sleep(10);
    double start_p = ncnn::get_current_time();
    double skp_time = 0;
    double pipe_time = 0;
    double upload_time=0;
    for (int i = 0; i < layer_count; i++)
    {
//        printf("c,%d,%d\n", sched_getcpu(), i);
        Layer* layer = d->layers[i];

        Option opt1 = opt;
#if NCNN_VULKAN
        if (opt.use_vulkan_compute)
        {
            if (!layer->support_image_storage) opt1.use_image_storage = false;
        }
        //upload_model
#endif // NCNN_VULKAN
        double starts = ncnn::get_current_time();

//        pthread_mutex_lock(&pipe_lock);
//        while (i  >= ncnn::current_layer_idx_f2p ){
//            pthread_cond_wait(&pipe_cond, &pipe_lock);
//        }
//        pthread_mutex_unlock(&pipe_lock);

        syn_wait(create_syn, i);

        double ends = ncnn::get_current_time();
        double times = ends - starts;
        skp_time+=times;
//        fprintf(stderr, "START pipe:  %s %d %d in pipeline time\n", d->layers[i]->name.c_str(), i-1, ncnn::current_layer_idx_f2p);
        if(i==1){
            pipe_start = ncnn::get_current_time();
        }
//        int cpu_ = sched_getcpu();
//        fseek(arm_weight_file_reads[cpu_], arm_weight_file_seek_Vectors[i-1], SEEK_SET);
        double start1 = ncnn::get_current_time();
//        printf("pipe %s\n", layer->name.c_str());
//        printf("%d\n", i);
        int cret = layer->create_pipeline(opt1);
//        printf("%s tid=%ld,cpu=%d\n", d->layers[i]->name.c_str(), pthread_self(), sched_getcpu());
//        std::cout<< typeid(layer).name()<<std::endl;
//        printf("%s\n", layer->name.c_str());
        double end1 = ncnn::get_current_time();

        if(USE_PACK_ARM&&ARM_W_TEST){
            //arm_weight_file_seek_save
            arm_weight_file_seek_Vectors[i] = arm_weight_file_seek_save;
//            if(arm_weight_file_seek_Vectors[i] != arm_weight_file_seek_Vectors[i-1]){
//                printf("%d,", i);
//            }
        }
//        if(i == 7)
//            printf("pipe time = %f\n", end1 - start1);

//        printf("%d,%d,%f,%s\n", sched_getcpu(), i, end1 - start1, layer->name.c_str());
//        syn_act(infer_syn, i);
        double time1 = end1 - start1;
        pipe_time += time1;

//        pthread_mutex_lock(&infer_lock);
//        ncnn::current_layer_idx_p2i =i+1;
//        pthread_mutex_unlock(&infer_lock);
//        pthread_cond_signal(&infer_cond);
        syn_act(infer_syn, i);
        //        fprintf(stderr, "%f\n", time1);
//        fprintf(stderr, "pipe:time %7.4f %s \n", time1,  d->layers[i]->name.c_str());
//        fprintf(stderr, "pipe:time %7.4f start %7.2f end: %7.2f %s %d %d \n", time1, start1-dr_start, end1-dr_start, d->layers[i]->name.c_str(), i, ncnn::current_layer_idx_f2p);
//        if(layer->name == "conv2/3x3")
//        fprintf(stderr, "%6.3f,", time1);
//        printf("%s\n",  d->layers[i]->name.c_str());
//        printf("%lu\n", sizeof(pipe_t_list)/ sizeof(pipe_t_list[0]));


        if (cret != 0)
        {
#if NCNN_STRING
            NCNN_LOGE("layer create_pipeline %d %s failed", i, layer->name.c_str());
#else
            NCNN_LOGE("layer create_pipeline %d failed", i);
#endif
            ret = -1;
            break;
        }
#if NCNN_VULKAN
        double svk = ncnn::get_current_time();
        if (opt.use_vulkan_compute)
        {
            ncnn::VkTransfer cmd(d->vkdev);

            // create gpu device allocator if null
            if (!d->weight_vkallocator)
            {
                d->weight_vkallocator = new VkWeightAllocator(d->vkdev);
            }
            if (!d->weight_staging_vkallocator)
            {
                d->weight_staging_vkallocator = new VkWeightStagingAllocator(d->vkdev);
            }

            Option opt_upload = opt;
            opt_upload.blob_vkallocator = d->weight_vkallocator;
            opt_upload.workspace_vkallocator = d->weight_vkallocator;
            opt_upload.staging_vkallocator = d->weight_staging_vkallocator;
            int uret = layer->upload_model(cmd, opt_upload);
            if (uret != 0)
            {
                NCNN_LOGE("layer upload_model %d failed", (int)i);
                return -1;
            }
        }
        double evk = ncnn::get_current_time();
        upload_time += evk - svk;
#endif //NCNN_VULKAN
    }
    printf("_____    skp_time=%f pipe_time=%f upload_time=%f\n", skp_time, pipe_time, upload_time);

    double end_p = ncnn::get_current_time();
    double time_p = end_p - start_p;
//    fprintf(stderr, "pipeline time %f\n", time_p);
    if (opt.use_local_pool_allocator)
    {
        if (opt.blob_allocator == 0)
        {
            if (!d->local_blob_allocator)
            {
                d->local_blob_allocator = new PoolAllocator;
                d->local_blob_allocator->set_size_compare_ratio(0.f);
            }
        }
        if (opt.workspace_allocator == 0)
        {
            if (!d->local_workspace_allocator)
            {
                d->local_workspace_allocator = new PoolAllocator;
                d->local_workspace_allocator->set_size_compare_ratio(0.5f);
            }
        }
    }

//#if NCNN_VULKAN
//    double upload_start = ncnn::get_current_time();
//    if (opt.use_vulkan_compute)
//    {
//        d->upload_model();
//    }
//    double upload_end = ncnn::get_current_time();
//    fprintf(stderr, "VK upload time %f\n", upload_end - upload_start);
//#endif // NCNN_VULKAN
    return ret;
}

int Net::load_model_pipe_cpu1()
{
    printf("_____    load_model_pipe_cpu1 tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());
    int ret = 0;
    int layer_count = (int)d->layers.size();
#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        if (!opt.pipeline_cache)
        {
            if (!d->pipeline_cache)
                d->pipeline_cache = new PipelineCache(d->vkdev);
            opt.pipeline_cache = d->pipeline_cache;
        }
    }
#endif // NCNN_VULKAN

       //    sleep(10);
    double start_p = ncnn::get_current_time();
    double skp_time = 0;
    double pipe_time = 0;
    for (int i = 0; i < 60; i++)
    {
        Layer* layer = d->layers[i];

        Option opt1 = opt;
#if NCNN_VULKAN
        if (opt.use_vulkan_compute)
        {
            if (!layer->support_image_storage) opt1.use_image_storage = false;
        }
#endif // NCNN_VULKAN
        pthread_mutex_lock(&pipe_lock);
        while (i  >= ncnn::current_layer_idx_f2p ){
            pthread_cond_wait(&pipe_cond, &pipe_lock);
            //            sleep(0);
        }
        pthread_mutex_unlock(&pipe_lock);

        if(i==0){
            pipe_start = ncnn::get_current_time();
            printf("----------------------start %f\n", pipe_start);
        }
        double start1 = ncnn::get_current_time();

        int cret = layer->create_pipeline(opt1);

        double end1 = ncnn::get_current_time();
        double time1 = end1 - start1;
        pipe_time += time1;

        pthread_mutex_lock(&infer_lock);
        ncnn::current_layer_idx_p2i =i+1;
        pthread_mutex_unlock(&infer_lock);
        pthread_cond_signal(&infer_cond);
        //        fprintf(stderr, "%f\n", time1);

        //        fprintf(stderr, "pipe:time %7.4f start %7.2f end: %7.2f %s %d %d \n", time1, start1-dr_start, end1-dr_start, d->layers[i]->name.c_str(), i, ncnn::current_layer_idx_f2p);
        //        fprintf(stderr, "%6.3f,", time1);
        //        printf("%s\n",  d->layers[i]->name.c_str());
        //        printf("%lu\n", sizeof(pipe_t_list)/ sizeof(pipe_t_list[0]));
        if (cret != 0)
        {
#if NCNN_STRING
            NCNN_LOGE("layer create_pipeline %d %s failed", i, layer->name.c_str());
#else
            NCNN_LOGE("layer create_pipeline %d failed", i);
#endif
            ret = -1;
            break;
        }
    }
    printf("skp_time=%f pipe_time=%f\n", skp_time, pipe_time);


    if (opt.use_local_pool_allocator)
    {
        if (opt.blob_allocator == 0)
        {
            if (!d->local_blob_allocator)
            {
                d->local_blob_allocator = new PoolAllocator;
                d->local_blob_allocator->set_size_compare_ratio(0.f);
            }
        }
        if (opt.workspace_allocator == 0)
        {
            if (!d->local_workspace_allocator)
            {
                d->local_workspace_allocator = new PoolAllocator;
                d->local_workspace_allocator->set_size_compare_ratio(0.5f);
            }
        }
    }

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        d->upload_model();
    }
#endif // NCNN_VULKAN

    return ret;
}
int Net::load_model_pipe_cpu2()
{
    printf("_____    load_model_pipe_cpu2 tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());
    int ret = 0;
    int layer_count = (int)d->layers.size();
#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        if (!opt.pipeline_cache)
        {
            if (!d->pipeline_cache)
                d->pipeline_cache = new PipelineCache(d->vkdev);
            opt.pipeline_cache = d->pipeline_cache;
        }
    }
#endif // NCNN_VULKAN

       //    sleep(10);
    double start_p = ncnn::get_current_time();
    double skp_time = 0;
    double pipe_time = 0;
    for (int i = 0; i < layer_count; i++)
    {
        if(i%2==0){
            continue;
        }
//        printf("pip %d\n", i);
        Layer* layer = d->layers[i];

        Option opt1 = opt;
#if NCNN_VULKAN
        if (opt.use_vulkan_compute)
        {
            if (!layer->support_image_storage) opt1.use_image_storage = false;
        }
#endif // NCNN_VULKAN

        pthread_mutex_lock(&pipe_lock);
        while (i  >= ncnn::current_layer_idx_f2p ){
            pthread_cond_wait(&pipe_cond, &pipe_lock);
            //            sleep(0);
        }
        pthread_mutex_unlock(&pipe_lock);

        if(i==0){
            pipe_start = ncnn::get_current_time();
        }
        double start1 = ncnn::get_current_time();

        int cret = layer->create_pipeline(opt1);

        double end1 = ncnn::get_current_time();
        double time1 = end1 - start1;
        pipe_time += time1;

        pthread_mutex_lock(&infer_lock);
        ncnn::current_layer_idx_p2i =i+1;
        pthread_mutex_unlock(&infer_lock);
        pthread_cond_signal(&infer_cond);
        //        fprintf(stderr, "%f\n", time1);

        //        fprintf(stderr, "pipe:time %7.4f start %7.2f end: %7.2f %s %d %d \n", time1, start1-dr_start, end1-dr_start, d->layers[i]->name.c_str(), i, ncnn::current_layer_idx_f2p);
        //        fprintf(stderr, "%6.3f,", time1);
        //        printf("%s\n",  d->layers[i]->name.c_str());
        //        printf("%lu\n", sizeof(pipe_t_list)/ sizeof(pipe_t_list[0]));
        if (cret != 0)
        {
#if NCNN_STRING
            NCNN_LOGE("layer create_pipeline %d %s failed", i, layer->name.c_str());
#else
            NCNN_LOGE("layer create_pipeline %d failed", i);
#endif
            ret = -1;
            break;
        }

        if(i==layer_count-1){
            pipe_end = ncnn::get_current_time();
            printf("------------------------pipe time ------------------%f\n", pipe_end-pipe_start);
        }
    }
    printf("skp_time=%f pipe_time=%f\n", skp_time, pipe_time);


    if (opt.use_local_pool_allocator)
    {
        if (opt.blob_allocator == 0)
        {
            if (!d->local_blob_allocator)
            {
                d->local_blob_allocator = new PoolAllocator;
                d->local_blob_allocator->set_size_compare_ratio(0.f);
            }
        }
        if (opt.workspace_allocator == 0)
        {
            if (!d->local_workspace_allocator)
            {
                d->local_workspace_allocator = new PoolAllocator;
                d->local_workspace_allocator->set_size_compare_ratio(0.5f);
            }
        }
    }

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        d->upload_model();
    }
#endif // NCNN_VULKAN

    return ret;
}
int Net::load_model_pipe_cpu3()
{
    printf("_____    load_model_pipe_cpu3 tid=%d,cpu=%d\n", pthread_self(), sched_getcpu());
    int ret = 0;
    int layer_count = (int)d->layers.size();
#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        if (!opt.pipeline_cache)
        {
            if (!d->pipeline_cache)
                d->pipeline_cache = new PipelineCache(d->vkdev);
            opt.pipeline_cache = d->pipeline_cache;
        }
    }
#endif // NCNN_VULKAN

       //    sleep(10);
    double start_p = ncnn::get_current_time();
    double skp_time = 0;
    double pipe_time = 0;
    for (int i = 0; i < layer_count; i++)
    {
        if(i%2!=0){
            continue;
        }
//        printf("pip %d\n", i);
        Layer* layer = d->layers[i];

        Option opt1 = opt;
#if NCNN_VULKAN
        if (opt.use_vulkan_compute)
        {
            if (!layer->support_image_storage) opt1.use_image_storage = false;
        }
#endif // NCNN_VULKAN

        pthread_mutex_lock(&pipe_lock);
        while (i  >= ncnn::current_layer_idx_f2p ){
            pthread_cond_wait(&pipe_cond, &pipe_lock);
            //            sleep(0);
        }
        pthread_mutex_unlock(&pipe_lock);

        if(i==0){
            pipe_start = ncnn::get_current_time();
        }
        double start1 = ncnn::get_current_time();

        int cret = layer->create_pipeline(opt1);

        double end1 = ncnn::get_current_time();
        double time1 = end1 - start1;
        pipe_time += time1;

//        pthread_mutex_lock(&infer_lock);
//        ncnn::current_layer_idx_p2i =i+1;
//        pthread_mutex_unlock(&infer_lock);
//        pthread_cond_signal(&infer_cond);
        //        fprintf(stderr, "%f\n", time1);

        //        fprintf(stderr, "pipe:time %7.4f start %7.2f end: %7.2f %s %d %d \n", time1, start1-dr_start, end1-dr_start, d->layers[i]->name.c_str(), i, ncnn::current_layer_idx_f2p);
        //        fprintf(stderr, "%6.3f,", time1);
        //        printf("%s\n",  d->layers[i]->name.c_str());
        //        printf("%lu\n", sizeof(pipe_t_list)/ sizeof(pipe_t_list[0]));
        if (cret != 0)
        {
#if NCNN_STRING
            NCNN_LOGE("layer create_pipeline %d %s failed", i, layer->name.c_str());
#else
            NCNN_LOGE("layer create_pipeline %d failed", i);
#endif
            ret = -1;
            break;
        }
        if(i==layer_count-1){
            pipe_end = ncnn::get_current_time();
            printf("------------------------pipe time ------------------%f\n", pipe_end-pipe_start);
        }
    }
    printf("skp_time=%f pipe_time=%f\n", skp_time, pipe_time);


    if (opt.use_local_pool_allocator)
    {
        if (opt.blob_allocator == 0)
        {
            if (!d->local_blob_allocator)
            {
                d->local_blob_allocator = new PoolAllocator;
                d->local_blob_allocator->set_size_compare_ratio(0.f);
            }
        }
        if (opt.workspace_allocator == 0)
        {
            if (!d->local_workspace_allocator)
            {
                d->local_workspace_allocator = new PoolAllocator;
                d->local_workspace_allocator->set_size_compare_ratio(0.5f);
            }
        }
    }

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        d->upload_model();
    }
#endif // NCNN_VULKAN

    return ret;
}
void Net::upload_models(){
#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        d->upload_model();
    }
#endif // NCNN_VULKAN
}

int Net::load_pipe_layer(int layer_idx)
{
    static float tp=0;
    int ret = 0;
//    int layer_count = (int)d->layers.size();
    Layer* layer = d->layers[layer_idx];

    Option opt1 = opt;
//    double start = get_current_time();
#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        if (!opt.pipeline_cache)
        {
            if (!d->pipeline_cache)
                d->pipeline_cache = new PipelineCache(d->vkdev);
            opt.pipeline_cache = d->pipeline_cache;
        }
    }
    if (opt.use_vulkan_compute)
    {
        if (!layer->support_image_storage) opt1.use_image_storage = false;
    }
#endif // NCNN_VULKAN


//    syn_wait(create_syn, index);

    double start_cp = ncnn::get_current_time();
    int cret = layer->create_pipeline(opt1);
    double end_cp = ncnn::get_current_time();

//    printf("+=====+trans %d [cpu%d] %s %f \n",  layer_idx, sched_getcpu(), layer->name.c_str(), end_cp - start_cp);
    if(save_time)
    {
        trans_starts[layer_idx] = (start_cp - save_start_time);
        trans_ends[layer_idx] = (end_cp - save_start_time);
    }
//    printf("=====create cpu%d %d %s\n", sched_getcpu(), layer_idx, layer->name.c_str());
//    printf("%d,%d,%f,%s\n", sched_getcpu(), layer_idx, end_cp - start_cp, layer->name.c_str());
//    tp += (end_cp - start_cp);
//    printf("tp, %f\n", tp);

//    syn_act(infer_syn, index);

    if (cret != 0)
    {
#if NCNN_STRING
        NCNN_LOGE("layer create_pipeline %d %s failed", layer_idx, layer->name.c_str());
#else
        NCNN_LOGE("layer create_pipeline %d failed", layer_idx);
#endif
        ret = -1;
    }
#if NCNN_VULKAN
    ////upload_model
//    if (opt.use_vulkan_compute)
//    {
//        ncnn::VkTransfer cmd(d->vkdev);
//
//        // create gpu device allocator if null
//        if (!d->weight_vkallocator)
//        {
//            d->weight_vkallocator = new VkWeightAllocator(d->vkdev);
//        }
//        if (!d->weight_staging_vkallocator)
//        {
//            d->weight_staging_vkallocator = new VkWeightStagingAllocator(d->vkdev);
//        }
//
//        Option opt_upload = opt;
//        opt_upload.blob_vkallocator = d->weight_vkallocator;
//        opt_upload.workspace_vkallocator = d->weight_vkallocator;
//        opt_upload.staging_vkallocator = d->weight_staging_vkallocator;
//        //    double svk = ncnn::get_current_time();
//        int uret = layer->upload_model(cmd, opt_upload);
//        if (uret != 0)
//        {
//            NCNN_LOGE("layer upload_model %d failed", (int)layer_idx);
//            return -1;
//        }
//    }
    double evk = ncnn::get_current_time();
#endif //NCNN_VULKAN

    if (opt.use_local_pool_allocator)
    {
        if (opt.blob_allocator == 0)
        {
            if (!d->local_blob_allocator)
            {
                d->local_blob_allocator = new PoolAllocator;
                d->local_blob_allocator->set_size_compare_ratio(0.f);
            }
        }
        if (opt.workspace_allocator == 0)
        {
            if (!d->local_workspace_allocator)
            {
                d->local_workspace_allocator = new PoolAllocator;
                d->local_workspace_allocator->set_size_compare_ratio(0.5f);
            }
        }
    }

//#if NCNN_VULKAN
//    if (opt.use_vulkan_compute)
//    {
//        d->upload_model();
//    }
//#endif // NCNN_VULKAN

//    double end = get_current_time();
//    printf("c,%d,%d\n", sched_getcpu(), layer_idx);
    return ret;
}

//Mutex upload_lock;
int Net::upload_model_layer(int layer_idx){
#if NCNN_VULKAN
    static float tt=0;
    double s = ncnn::get_current_time();
//    MutexLockGuard lock(upload_lock);
//    upload_lock.lock();
    Layer* layer = d->layers[layer_idx];
    if (opt.use_vulkan_compute)
    {
//        printf("=====upload cpu%d %d %s\n", sched_getcpu(), layer_idx, layer->name.c_str());
        static ncnn::VkTransfer cmd(d->vkdev);
//        printf("%d\n", &cmd);

        // create gpu device allocator if null
        if (!d->weight_vkallocator)
        {
            d->weight_vkallocator = new VkWeightAllocator(d->vkdev);
        }
        if (!d->weight_staging_vkallocator)
        {
            d->weight_staging_vkallocator = new VkWeightStagingAllocator(d->vkdev);
        }

        Option opt_upload = opt;
        opt_upload.blob_vkallocator = d->weight_vkallocator;
        opt_upload.workspace_vkallocator = d->weight_vkallocator;
        opt_upload.staging_vkallocator = d->weight_staging_vkallocator;
        //    double svk = ncnn::get_current_time();
//        upload_lock.lock();
        int uret = layer->upload_model(cmd, opt_upload);
//        upload_lock.unlock();
        if (uret != 0)
        {
            NCNN_LOGE("layer upload_model %d failed", (int)layer_idx);
            return -1;
        }

//        printf("waie\n");
//        cmd.submit_and_wait();
    }

    double e = ncnn::get_current_time();
    tt+= (e-s);
//    printf("upload %d %f %f\n", layer_idx, e-s, tt);
//    upload_lock.unlock();
    return 0;
#endif // NCNN_VULKAN
}
#if NCNN_STDIO
#if NCNN_STRING
int Net::load_param(FILE* fp)
{
    DataReaderFromStdio dr(fp);
    return load_param(dr);
}

int Net::load_param_mem(const char* _mem)
{
    const unsigned char* mem = (const unsigned char*)_mem;
    DataReaderFromMemory dr(mem);
    return load_param(dr);
}

int Net::load_param(const char* protopath)
{
    FILE* fp = fopen(protopath, "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen %s failed", protopath);
        return -1;
    }

    int ret = load_param(fp);
    fclose(fp);
    return ret;
}
#endif // NCNN_STRING

int Net::load_param_bin(FILE* fp)
{
    DataReaderFromStdio dr(fp);
    return load_param_bin(dr);
}

int Net::load_param_bin(const char* protopath)
{
    FILE* fp = fopen(protopath, "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen %s failed", protopath);
        return -1;
    }

    int ret = load_param_bin(fp);
    fclose(fp);
    return ret;
}

int Net::load_model(FILE* fp)
{
    DataReaderFromStdio dr(fp);
    return load_model(dr);
}

int Net::load_model(const char* modelpath)
{
    FILE* fp = fopen(modelpath, "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen %s failed", modelpath);
        return -1;
    }

    int ret = load_model(fp);
    fclose(fp);
    return ret;
}
#endif // NCNN_STDIO

int Net::load_param(const unsigned char* _mem)
{
    const unsigned char* mem = _mem;
    DataReaderFromMemory dr(mem);
    load_param_bin(dr);
    return static_cast<int>(mem - _mem);
}

int Net::load_model(const unsigned char* _mem)
{
    const unsigned char* mem = _mem;
    DataReaderFromMemory dr(mem);
    load_model(dr);
    return static_cast<int>(mem - _mem);
}

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 9
#if NCNN_STRING
int Net::load_param(AAsset* asset)
{
    DataReaderFromAndroidAsset dr(asset);
    return load_param(dr);
}

int Net::load_param(AAssetManager* mgr, const char* assetpath)
{
    AAsset* asset = AAssetManager_open(mgr, assetpath, AASSET_MODE_BUFFER);
    if (!asset)
    {
        NCNN_LOGE("AAssetManager_open %s failed", assetpath);
        return -1;
    }

    int ret = load_param(asset);
    AAsset_close(asset);
    return ret;
}
#endif // NCNN_STRING

int Net::load_param_bin(AAsset* asset)
{
    DataReaderFromAndroidAsset dr(asset);
    return load_param_bin(dr);
}

int Net::load_param_bin(AAssetManager* mgr, const char* assetpath)
{
    AAsset* asset = AAssetManager_open(mgr, assetpath, AASSET_MODE_BUFFER);
    if (!asset)
    {
        NCNN_LOGE("AAssetManager_open %s failed", assetpath);
        return -1;
    }

    int ret = load_param_bin(asset);
    AAsset_close(asset);
    return ret;
}

int Net::load_model(AAsset* asset)
{
    DataReaderFromAndroidAsset dr(asset);
    return load_model(dr);
}

int Net::load_model(AAssetManager* mgr, const char* assetpath)
{
    AAsset* asset = AAssetManager_open(mgr, assetpath, AASSET_MODE_STREAMING);
    if (!asset)
    {
        NCNN_LOGE("AAssetManager_open %s failed", assetpath);
        return -1;
    }

    int ret = load_model(asset);
    AAsset_close(asset);
    return ret;
}
#endif // __ANDROID_API__ >= 9
#endif // NCNN_PLATFORM_API

void Net::clear()
{
    d->blobs.clear();
    for (size_t i = 0; i < d->layers.size(); i++)
    {
        Layer* layer = d->layers[i];

        Option opt1 = opt;
        if (!layer->support_image_storage)
        {
            opt1.use_image_storage = false;
        }

        int dret = layer->destroy_pipeline(opt1);
        if (dret != 0)
        {
            NCNN_LOGE("layer destroy_pipeline failed");
            // ignore anyway
        }

        if (layer->typeindex & ncnn::LayerType::CustomBit)
        {
            int custom_index = layer->typeindex & ~ncnn::LayerType::CustomBit;
            if (d->custom_layer_registry[custom_index].destroyer)
            {
                d->custom_layer_registry[custom_index].destroyer(layer, d->custom_layer_registry[custom_index].userdata);
            }
            else
            {
                delete layer;
            }
        }
        else
        {
            delete layer;
        }
    }
    d->layers.clear();

    if (d->local_blob_allocator)
    {
        delete d->local_blob_allocator;
        d->local_blob_allocator = 0;
    }
    if (d->local_workspace_allocator)
    {
        delete d->local_workspace_allocator;
        d->local_workspace_allocator = 0;
    }

#if NCNN_VULKAN
    if (d->weight_vkallocator)
    {
        delete d->weight_vkallocator;
        d->weight_vkallocator = 0;
    }
    if (d->weight_staging_vkallocator)
    {
        delete d->weight_staging_vkallocator;
        d->weight_staging_vkallocator = 0;
    }
    if (d->pipeline_cache)
    {
        delete d->pipeline_cache;
        d->pipeline_cache = 0;
        opt.pipeline_cache = 0;
    }
#endif // NCNN_VULKAN
}

Extractor Net::create_extractor() const
{
    return Extractor(this, d->blobs.size());
}

const std::vector<int>& Net::input_indexes() const
{
    return d->input_blob_indexes;
}

const std::vector<int>& Net::output_indexes() const
{
    return d->output_blob_indexes;
}

#if NCNN_STRING
const std::vector<const char*>& Net::input_names() const
{
    return d->input_blob_names;
}

const std::vector<const char*>& Net::output_names() const
{
    return d->output_blob_names;
}
#endif

const std::vector<Blob>& Net::blobs() const
{
    return d->blobs;
}

const std::vector<Layer*>& Net::layers() const
{
    return d->layers;
}

std::vector<Blob>& Net::mutable_blobs()
{
    return d->blobs;
}

std::vector<Layer*>& Net::mutable_layers()
{
    return d->layers;
}

#if NCNN_VULKAN
void Net::set_vulkan_device(int device_index)
{
    d->vkdev = get_gpu_device(device_index);
}

void Net::set_vulkan_device(const VulkanDevice* _vkdev)
{
    d->vkdev = _vkdev;
}

const VulkanDevice* Net::vulkan_device() const
{
    return d->vkdev;
}
#endif // NCNN_VULKAN

#if NCNN_STRING
int Net::find_blob_index_by_name(const char* name) const
{
    for (size_t i = 0; i < d->blobs.size(); i++)
    {
        const Blob& blob = d->blobs[i];
        if (blob.name == name)
        {
            return static_cast<int>(i);
        }
    }

    NCNN_LOGE("find_blob_index_by_name %s failed", name);
    return -1;
}

int Net::find_layer_index_by_name(const char* name) const
{
    for (size_t i = 0; i < d->layers.size(); i++)
    {
        const Layer* layer = d->layers[i];
        if (layer->name == name)
        {
            return static_cast<int>(i);
        }
    }

    NCNN_LOGE("find_layer_index_by_name %s failed", name);
    return -1;
}

int Net::custom_layer_to_index(const char* type)
{
    const size_t custom_layer_registry_entry_count = d->custom_layer_registry.size();
    for (size_t i = 0; i < custom_layer_registry_entry_count; i++)
    {
        if (strcmp(type, d->custom_layer_registry[i].name) == 0)
            return static_cast<int>(i);
    }

    return -1;
}

Layer* Net::create_custom_layer(const char* type)
{
    int index = custom_layer_to_index(type);
    if (index == -1)
        return 0;

    return create_custom_layer(index);
}
#endif // NCNN_STRING

Layer* Net::create_custom_layer(int index)
{
    const size_t custom_layer_registry_entry_count = d->custom_layer_registry.size();
    if (index < 0 || static_cast<unsigned int>(index) >= custom_layer_registry_entry_count)
        return 0;

    layer_creator_func layer_creator = d->custom_layer_registry[index].creator;
    if (!layer_creator)
        return 0;

    Layer* layer = layer_creator(d->custom_layer_registry[index].userdata);
    layer->typeindex = ncnn::LayerType::CustomBit | index;
    return layer;
}

class ExtractorPrivate
{
public:
    ExtractorPrivate(const Net* _net)
        : net(_net)
    {
    }
    const Net* net;
    std::vector<Mat> blob_mats;
    Option opt;

#if NCNN_VULKAN
    VkAllocator* local_blob_vkallocator;
    VkAllocator* local_staging_vkallocator;

    std::vector<VkMat> blob_mats_gpu;
    std::vector<VkImageMat> blob_mats_gpu_image;
#endif // NCNN_VULKAN
};

Extractor::Extractor(const Net* _net, size_t blob_count)
    : d(new ExtractorPrivate(_net))
{
    d->blob_mats.resize(blob_count);
    d->opt = d->net->opt;

#if NCNN_VULKAN
    if (d->net->opt.use_vulkan_compute)
    {
        d->local_blob_vkallocator = 0;
        d->local_staging_vkallocator = 0;

        d->blob_mats_gpu.resize(blob_count);
        d->blob_mats_gpu_image.resize(blob_count);
    }
#endif // NCNN_VULKAN
}

Extractor::~Extractor()
{
    clear();

    delete d;
}

Extractor::Extractor(const Extractor& rhs)
    : d(new ExtractorPrivate(0))
{
    d->net = rhs.d->net;
    d->blob_mats = rhs.d->blob_mats;
    d->opt = rhs.d->opt;

#if NCNN_VULKAN
    d->local_blob_vkallocator = 0;
    d->local_staging_vkallocator = 0;

    d->blob_mats_gpu = rhs.d->blob_mats_gpu;
    d->blob_mats_gpu_image = rhs.d->blob_mats_gpu_image;
#endif // NCNN_VULKAN
}

Extractor& Extractor::operator=(const Extractor& rhs)
{
    if (this == &rhs)
        return *this;

    d->net = rhs.d->net;
    d->blob_mats = rhs.d->blob_mats;
    d->opt = rhs.d->opt;

#if NCNN_VULKAN
    d->local_blob_vkallocator = 0;
    d->local_staging_vkallocator = 0;

    d->blob_mats_gpu = rhs.d->blob_mats_gpu;
    d->blob_mats_gpu_image = rhs.d->blob_mats_gpu_image;
#endif // NCNN_VULKAN

    return *this;
}

void Extractor::clear()
{
    d->blob_mats.clear();

#if NCNN_VULKAN
    if (d->opt.use_vulkan_compute)
    {
        d->blob_mats_gpu.clear();
        d->blob_mats_gpu_image.clear();

        if (d->local_blob_vkallocator)
        {
            d->net->vulkan_device()->reclaim_blob_allocator(d->local_blob_vkallocator);
        }
        if (d->local_staging_vkallocator)
        {
            d->net->vulkan_device()->reclaim_staging_allocator(d->local_staging_vkallocator);
        }
    }
#endif // NCNN_VULKAN
}

void Extractor::set_light_mode(bool enable)
{
    d->opt.lightmode = enable;
}

void Extractor::set_num_threads(int num_threads)
{
    d->opt.num_threads = num_threads;
}

void Extractor::set_blob_allocator(Allocator* allocator)
{
    d->opt.blob_allocator = allocator;
}

void Extractor::set_workspace_allocator(Allocator* allocator)
{
    d->opt.workspace_allocator = allocator;
}

#if NCNN_VULKAN
void Extractor::set_vulkan_compute(bool enable)
{
    if (d->net->d->opt.use_vulkan_compute)
    {
        d->opt.use_vulkan_compute = enable;
    }
    else
    {
        NCNN_LOGE("set_vulkan_compute failed, network use_vulkan_compute disabled");
    }
}

void Extractor::set_blob_vkallocator(VkAllocator* allocator)
{
    d->opt.blob_vkallocator = allocator;
}

void Extractor::set_workspace_vkallocator(VkAllocator* allocator)
{
    d->opt.workspace_vkallocator = allocator;
}

void Extractor::set_staging_vkallocator(VkAllocator* allocator)
{
    d->opt.staging_vkallocator = allocator;
}
#endif // NCNN_VULKAN

#if NCNN_STRING
int Extractor::input(const char* blob_name, const Mat& in)
{
    int blob_index = d->net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
    {
        NCNN_LOGE("Try");
        const std::vector<const char*>& input_names = d->net->input_names();
        for (size_t i = 0; i < input_names.size(); i++)
        {
            NCNN_LOGE("    ex.input(\"%s\", in%d);", input_names[i], (int)i);
        }

        return -1;
    }

    return input(blob_index, in);
}

int Extractor::extract(const char* blob_name, Mat& feat, int type)
{
    int blob_index = d->net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
    {
        NCNN_LOGE("Try");
        const std::vector<const char*>& output_names = d->net->output_names();
        for (size_t i = 0; i < output_names.size(); i++)
        {
            NCNN_LOGE("    ex.extract(\"%s\", out%d);", output_names[i], (int)i);
        }

        return -1;
    }
    return extract(blob_index, feat, type);
}
#endif // NCNN_STRING

int Extractor::input(int blob_index, const Mat& in)
{
    if (blob_index < 0 || blob_index >= (int)d->blob_mats.size())
        return -1;

    d->blob_mats[blob_index] = in;

    return 0;
}

int Extractor::extract(int blob_index, Mat& feat, int type)
{
    if (blob_index < 0 || blob_index >= (int)d->blob_mats.size())
        return -1;

    int old_blocktime = get_kmp_blocktime();
    set_kmp_blocktime(d->opt.openmp_blocktime);

    int old_flush_denormals = get_flush_denormals();
    set_flush_denormals(d->opt.flush_denormals);

    int ret = 0;

    if (d->blob_mats[blob_index].dims == 0)
    {
        int layer_index = d->net->blobs()[blob_index].producer;

        // use local allocator
        if (d->opt.use_local_pool_allocator)
        {
            if (!d->opt.blob_allocator)
            {
                d->opt.blob_allocator = d->net->d->local_blob_allocator;
            }
            if (!d->opt.workspace_allocator)
            {
                d->opt.workspace_allocator = d->net->d->local_workspace_allocator;
            }
        }

#if NCNN_VULKAN
        if (d->opt.use_vulkan_compute)
        {
            // use local allocator
            if (!d->opt.blob_vkallocator)
            {
                d->local_blob_vkallocator = d->net->vulkan_device()->acquire_blob_allocator();
                d->opt.blob_vkallocator = d->local_blob_vkallocator;
            }
            if (!d->opt.workspace_vkallocator)
            {
                d->opt.workspace_vkallocator = d->opt.blob_vkallocator;
            }
            if (!d->opt.staging_vkallocator)
            {
                d->local_staging_vkallocator = d->net->vulkan_device()->acquire_staging_allocator();
                d->opt.staging_vkallocator = d->local_staging_vkallocator;
            }

            ncnn::VkCompute cmd(d->net->vulkan_device());
#if NCNN_BENCHMARK
            cmd.create_query_pool(d->net->layers().size() * 2);
#endif // NCNN_BENCHMARK

            // TODO vkimagemat for adreno
            if (d->opt.use_image_storage)
            {
                VkImageMat feat_gpu;
                ret = extract(blob_index, feat_gpu, cmd);

                if (d->blob_mats[blob_index].dims == 0 && feat_gpu.dims != 0)
                {
                    cmd.record_download(feat_gpu, d->blob_mats[blob_index], d->opt);

                    cmd.submit_and_wait();

#if NCNN_BENCHMARK
                    std::vector<uint64_t> results(d->net->layers().size() * 2);
                    cmd.get_query_pool_results(0, d->net->layers().size() * 2, results);
                    for (size_t i = 0; i < d->net->layers().size(); i++)
                    {
                        uint64_t start = results[i * 2];
                        uint64_t end = results[i * 2 + 1];
                        if (start == 0 || end == 0)
                            continue;

                        double duration_us = (end - start) * d->net->vulkan_device()->info.timestamp_period() / 1000;
                        NCNN_LOGE("%-24s %-30s %8.2lfus    |", d->net->layers()[i]->type.c_str(), d->net->layers()[i]->name.c_str(), duration_us);
                    }
#endif // NCNN_BENCHMARK
                }
            }
            else
            {
                VkMat feat_gpu;
                ret = extract(blob_index, feat_gpu, cmd);

                if (d->blob_mats[blob_index].dims == 0 && feat_gpu.dims != 0)
                {

                    double st = ncnn::get_current_time();
                    cmd.record_download(feat_gpu, d->blob_mats[blob_index], d->opt);
                    double et = ncnn::get_current_time();
//                    printf("++++++++++++++++++++ cmd.record_download %f \n", et-st);

                    cmd.submit_and_wait();
                    double e = ncnn::get_current_time();
//                    printf("--------------------- cmd.submit_and_wait %f\n", e-et);

#if NCNN_BENCHMARK
                    std::vector<uint64_t> results(d->net->layers().size() * 2);
                    cmd.get_query_pool_results(0, d->net->layers().size() * 2, results);
                    for (size_t i = 0; i < d->net->layers().size(); i++)
                    {
                        uint64_t start = results[i * 2];
                        uint64_t end = results[i * 2 + 1];
                        if (start == 0 || end == 0)
                            continue;

                        double duration_us = (end - start) * d->net->vulkan_device()->info.timestamp_period() / 1000;
                        NCNN_LOGE("%-24s %-30s %8.2lfus    |", d->net->layers()[i]->type.c_str(), d->net->layers()[i]->name.c_str(), duration_us);
                    }
#endif // NCNN_BENCHMARK
                }
            }
        }
        else
        {
//            double s = ncnn::get_current_time();
            ret = d->net->d->forward_layer(layer_index, d->blob_mats, d->opt);
//            double e = ncnn::get_current_time();
//            printf("oouutt  %f\n", e-s);
        }
#else
        ret = d->net->d->forward_layer(layer_index, d->blob_mats, d->opt);
#endif // NCNN_VULKAN
    }

    feat = d->blob_mats[blob_index];

    if (d->opt.use_packing_layout && (type == 0) && feat.elempack != 1)
    {
        Mat bottom_blob_unpacked;
        convert_packing(feat, bottom_blob_unpacked, 1, d->opt);
        feat = bottom_blob_unpacked;
    }

    // clang-format off
    // *INDENT-OFF*
#if NCNN_ARM82
    if (d->opt.use_fp16_storage && cpu_support_arm_asimdhp() && (type == 0))
    {
        if (feat.elembits() == 16)
        {
            Mat feat_fp32;
            cast_float16_to_float32(feat, feat_fp32, d->opt);
            feat = feat_fp32;
        }
    }
    else
#endif // NCNN_ARM82
    if (d->opt.use_bf16_storage && (type == 0))
    {
        if (feat.elembits() == 16)
        {
            Mat feat_fp32;
            cast_bfloat16_to_float32(feat, feat_fp32, d->opt);
            feat = feat_fp32;
        }
    }
    else if (feat.elembits() == 8 && (type == 0))
    {
        Mat feat_fp32;
        cast_int8_to_float32(feat, feat_fp32, d->opt);
        feat = feat_fp32;
    }
    // *INDENT-ON*
    // clang-format on

    set_kmp_blocktime(old_blocktime);
    set_flush_denormals(old_flush_denormals);

    return ret;
}

#if NCNN_VULKAN
#if NCNN_STRING
int Extractor::input(const char* blob_name, const VkMat& in)
{
    int blob_index = d->net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
    {
        NCNN_LOGE("Try");
        const std::vector<const char*>& input_names = d->net->input_names();
        for (size_t i = 0; i < input_names.size(); i++)
        {
            NCNN_LOGE("    ex.input(\"%s\", in%d);", input_names[i], (int)i);
        }

        return -1;
    }

    return input(blob_index, in);
}

int Extractor::extract(const char* blob_name, VkMat& feat, VkCompute& cmd)
{
    int blob_index = d->net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
    {
        NCNN_LOGE("Try");
        const std::vector<const char*>& output_names = d->net->output_names();
        for (size_t i = 0; i < output_names.size(); i++)
        {
            NCNN_LOGE("    ex.extract(\"%s\", out%d);", output_names[i], (int)i);
        }

        return -1;
    }

    return extract(blob_index, feat, cmd);
}

int Extractor::input(const char* blob_name, const VkImageMat& in)
{
    int blob_index = d->net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
    {
        NCNN_LOGE("Try");
        const std::vector<const char*>& input_names = d->net->input_names();
        for (size_t i = 0; i < input_names.size(); i++)
        {
            NCNN_LOGE("    ex.input(\"%s\", in%d);", input_names[i], (int)i);
        }

        return -1;
    }

    return input(blob_index, in);
}

int Extractor::extract(const char* blob_name, VkImageMat& feat, VkCompute& cmd)
{
    int blob_index = d->net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
    {
        NCNN_LOGE("Try");
        const std::vector<const char*>& output_names = d->net->output_names();
        for (size_t i = 0; i < output_names.size(); i++)
        {
            NCNN_LOGE("    ex.extract(\"%s\", out%d);", output_names[i], (int)i);
        }

        return -1;
    }

    return extract(blob_index, feat, cmd);
}
#endif // NCNN_STRING

int Extractor::input(int blob_index, const VkMat& in)
{
    if (blob_index < 0 || blob_index >= (int)d->blob_mats.size())
        return -1;

    d->blob_mats_gpu[blob_index] = in;

    return 0;
}

int Extractor::extract(int blob_index, VkMat& feat, VkCompute& cmd)
{
    if (blob_index < 0 || blob_index >= (int)d->blob_mats.size())
        return -1;

    int old_blocktime = get_kmp_blocktime();
    set_kmp_blocktime(d->opt.openmp_blocktime);

    int old_flush_denormals = get_flush_denormals();
    set_flush_denormals(d->opt.flush_denormals);

    int ret = 0;

    if (d->blob_mats_gpu[blob_index].dims == 0)
    {
        if (d->blob_mats_gpu_image[blob_index].dims != 0)
        {
            // image to buffer
            cmd.record_image_to_buffer(d->blob_mats_gpu_image[blob_index], d->blob_mats_gpu[blob_index], d->opt);
        }
        else if (d->blob_mats[blob_index].dims != 0)
        {
            // host to buffer
            cmd.record_upload(d->blob_mats[blob_index], d->blob_mats_gpu[blob_index], d->opt);
        }
        else
        {
            int layer_index = d->net->blobs()[blob_index].producer;
            double s= ncnn::get_current_time();
            ret = d->net->d->forward_layer(layer_index, d->blob_mats, d->blob_mats_gpu, cmd, d->opt);
            double e = ncnn::get_current_time();
            printf("===================== forward_layer %f\n", e-s);
        }
    }

    feat = d->blob_mats_gpu[blob_index];

    set_kmp_blocktime(old_blocktime);
    set_flush_denormals(old_flush_denormals);

    return ret;
}

int Extractor::input(int blob_index, const VkImageMat& in)
{
    if (blob_index < 0 || blob_index >= (int)d->blob_mats.size())
        return -1;

    d->blob_mats_gpu_image[blob_index] = in;

    return 0;
}

int Extractor::extract(int blob_index, VkImageMat& feat, VkCompute& cmd)
{
    if (blob_index < 0 || blob_index >= (int)d->blob_mats.size())
        return -1;

    int old_blocktime = get_kmp_blocktime();
    set_kmp_blocktime(d->opt.openmp_blocktime);

    int old_flush_denormals = get_flush_denormals();
    set_flush_denormals(d->opt.flush_denormals);

    int ret = 0;

    if (d->blob_mats_gpu_image[blob_index].dims == 0)
    {
        if (d->blob_mats_gpu[blob_index].dims != 0)
        {
            // buffer to image
            cmd.record_buffer_to_image(d->blob_mats_gpu[blob_index], d->blob_mats_gpu_image[blob_index], d->opt);
        }
        else if (d->blob_mats[blob_index].dims != 0)
        {
            // host to image
            cmd.record_upload(d->blob_mats[blob_index], d->blob_mats_gpu_image[blob_index], d->opt);
        }
        else
        {
            int layer_index = d->net->blobs()[blob_index].producer;
            ret = d->net->d->forward_layer(layer_index, d->blob_mats, d->blob_mats_gpu, d->blob_mats_gpu_image, cmd, d->opt);
        }
    }

    feat = d->blob_mats_gpu_image[blob_index];

    if (feat.empty())
    {
        NCNN_LOGE("extract %d image allocation failed", blob_index);
        ret = -100;
    }

    set_kmp_blocktime(old_blocktime);
    set_flush_denormals(old_flush_denormals);

    return ret;
}
#endif // NCNN_VULKAN

} // namespace ncnn

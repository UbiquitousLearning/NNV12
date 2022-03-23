// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <float.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <algorithm>
#include <iostream>
#include <utility>
#include <thread>
#include <chrono>
#include <functional>
#include <atomic>
#include <sys/time.h>
#include <omp.h>
#include <fstream>

#ifdef _WIN32
#include <algorithm>
#include <windows.h> // Sleep()
#else
#include <unistd.h> // sleep()
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "net_new.h"
#include "gpu.h"
#include "pipeline.h"

#define __ARM_NEON 0
#define __CPU_MASK__ 1
#if __ARM_NEON
#include <arm_neon.h>
#endif

//
//export GOMP_CPU_AFFINITY="7 6 5 4";

char globalbinpath[256];
char gloableparampath[256];
#define THR 1
//#define ONE 1
//#define TWO 1
//#define FOR 1
size_t dr_cpu=2;//5;
size_t pipe_cpu=1;//6;
size_t infer_cpu=0;//0;
int warmUp = 0;

double start_t = 0;

//ncnn::Net net;
#ifdef __EMSCRIPTEN__
#define MODEL_DIR "/working/"
#else
#define MODEL_DIR ""
#endif

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* format, void* p) const
    {
        return 0;
    }
    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

static int g_warmup_loop_count = 8;
static int g_loop_count = 4;
static bool g_enable_cooling_down = true;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

#if NCNN_VULKAN
static ncnn::VulkanDevice* g_vkdev = 0;
static ncnn::VkAllocator* g_blob_vkallocator = 0;
static ncnn::VkAllocator* g_staging_vkallocator = 0;
#endif // NCNN_VULKAN

void benchmark(const char* comment, const ncnn::Mat& _in, const ncnn::Option& opt)
{
    int warmUp__ = 0;
    if(warmUp__)
    {
        ncnn::Net net_in;
        net_in.opt = opt;
        char tparampath[256];
        sprintf(tparampath, MODEL_DIR "mobilenet_v3.param"); //GoogleNet  //AlexNet
        char tbinpath[256];
        sprintf(tbinpath, MODEL_DIR "mobilenet_v3.bin");
        net_in.load_param(tparampath);
        int open = net_in.load_model_dr(tbinpath);
        if (open < 0)
        {
            printf("load file files\n");
            DataReaderFromEmpty dr;
            net_in.load_model_dr(dr);
        }
        net_in.load_model_pipe();
        //    printf("load p end %f\n", ncnn::get_current_time()-start_t);
        ncnn::Mat in_ = _in;
        in_.fill(0.01f);
        const std::vector<const char*>& input_names = net_in.input_names();
        const std::vector<const char*>& output_names = net_in.output_names();
        ncnn::Mat out_;
        for (int i = 0; i < 30; i++)
        {
            ncnn::Mat top;
            //        conv7x7s2_pack1to4_neon(ncnn::Mat(227, 227, 3), top, ncnn::Mat(7,7, 3), ncnn::Mat(7,7, 3), net->opt);
            //            inference(net_in, ncnn::Mat(227, 227, 3), false);
            //        ncnn::do_forward_layer
            ncnn::Extractor ex = net_in.create_extractor();
            ex.input(input_names[0], in_);
            ex.extract(output_names[0], out_);

            //        printf("infer p end %f\n", ncnn::get_current_time()-start_t);
        }
    }

    ncnn::Mat in = _in;
    in.fill(0.01f);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        g_blob_vkallocator->clear();
        g_staging_vkallocator->clear();
    }
#endif // NCNN_VULKAN

    ncnn::Net net;

    net.opt = opt;

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        net.set_vulkan_device(g_vkdev);
    }
#endif // NCNN_VULKAN

#ifdef __EMSCRIPTEN__
#define MODEL_DIR "/working/"
#else
#define MODEL_DIR ""
#endif

    char parampath[256];
    sprintf(parampath, MODEL_DIR "%s.param", comment);

    char binpath[256];
    sprintf(binpath, MODEL_DIR "%s.bin", comment);

    double load_param_start = ncnn::get_current_time();
    net.load_param(parampath);
    double load_param_end = ncnn::get_current_time();
    double load_param_time = load_param_end - load_param_start;

    double load_model_start = ncnn::get_current_time();

//    int open = net.load_model_dr(binpath);
//    if (open<0)
//    {
//        DataReaderFromEmpty dr;
//        net.load_model_dr(dr);
//    }
//    double load_model_end = ncnn::get_current_time();
//    double load_model_time = load_model_end - load_model_start;
//    printf("load model -------------------%f\n", ncnn::malloc_time);
//
//
//    net.load_model_pipe();
//    double load_pipe_end = ncnn::get_current_time();
//    double load_pipe_time = load_pipe_end - load_model_end;
//    printf("load pipe -------------------%f\n", ncnn::malloc_time);

    //    DataReaderFromEmpty dr;
    //    net.load_model(dr);


    FILE* fp = fopen(binpath, "rb");
    ncnn::DataReaderFromStdio dr(fp);
    ncnn::ModelBinFromDataReader mb(dr);
    printf("====================\n");
    for(int i=0; i<net.layers().size(); i++){
        printf("layer %d\n", i);
        net.load_model_layer(mb, i);
        net.load_pipe_layer(i);
        net.upload_model_layer(i);
        syn_act(infer_syn, i);
    }

    double load_model_end = ncnn::get_current_time();
    double load_model_time = load_model_end - load_model_start;


    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    if (g_enable_cooling_down)
    {
        // sleep 10 seconds for cooling down SOC  :(
#ifdef _WIN32
        Sleep(10 * 1000);
#elif defined(__unix__) || defined(__APPLE__)
        // sleep(10);
#elif _POSIX_TIMERS
        struct timespec ts;
        ts.tv_sec = 10;
        ts.tv_nsec = 0;
        nanosleep(&ts, &ts);
#else
        // TODO How to handle it ?
#endif
    }

    ncnn::Mat out;

    // warm up
    double first_time;
    for (int i = 0; i < 1; i++)
    {
        double start = ncnn::get_current_time();

        ncnn::Extractor ex = net.create_extractor();
        ex.input(input_names[0], in);
        ex.extract(output_names[0], out);

        double end = ncnn::get_current_time();
        first_time = end - start;
        //        if (i==0){
        //            first_time = time;
        //        }
        //                fprintf(stderr, "first %f\n", for_cal_time);
    }
    printf("inference -------------------%f\n", ncnn::malloc_time);

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i = 0; i < g_loop_count; i++)
    {
        for_cal_time = 0;
        double start = ncnn::get_current_time();

        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input(input_names[0], in);
            ex.extract(output_names[0], out);
            float* ptr = (float*)out.data;
            printf("out %f\n", *ptr);
            //                        fprintf(stderr, "%f\n", for_cal_time);
        }

        double end = ncnn::get_current_time();

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
        //        fprintf(stderr, "%f\n", time);
    }

    time_avg /= g_loop_count;

    fprintf(stderr, "%20s  load param = %7.2f  load model = %7.2f  first = %7.2f  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, load_param_time, load_model_time, first_time, time_min, time_max, time_avg);
}

void inference(const ncnn::Net& net, const ncnn::Mat& in, bool print=true){
    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    if (g_enable_cooling_down)
    {
        // sleep 10 seconds for cooling down SOC  :(
#ifdef _WIN32
        Sleep(10 * 1000);
#elif defined(__unix__) || defined(__APPLE__)
//    sleep(10);
#elif _POSIX_TIMERS
        struct timespec ts;
        ts.tv_sec = 10;
        ts.tv_nsec = 0;
        nanosleep(&ts, &ts);
#else
// TODO How to handle it ?
#endif
    }

    ncnn::Mat out;

    //    infer_start = ncnn::get_current_time();

    ncnn::Extractor ex = net.create_extractor();
    ex.input(input_names[0], in);
    ex.extract(output_names[0], out);

    float* ptr = (float*)out.data;
    if(print)
    {
        printf("%f\n", *ptr);
        infer_end = ncnn::get_current_time();
        infer_time = infer_end - infer_start;
        printf("for_skp_time=%f for_cal_time=%f\n", for_skp_time, for_cal_time);
        fprintf(stderr, "infer time =  %7.2f\n", infer_time);
    }

    //    pthread_exit(&t_infer);

}

void WriteSprivMapBinaryFile( const char* comment)
{
    char path[256];
    sprintf(path, MODEL_DIR "%s.m.dat", comment);
    std::ofstream osData(path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

    int ssSize = ncnn::spriv_map.size();
    osData.write(reinterpret_cast<char *>(&ssSize), sizeof(ssSize));
    for(auto &it:ncnn::spriv_map)
    {
        auto k = it.first;
        osData.write(reinterpret_cast<char *>(&k), sizeof(k));
        auto s = it.second;
        int sSize = s.size();
        osData.write(reinterpret_cast<char *>(&sSize), sizeof(sSize));
        osData.write(reinterpret_cast<char *>(s.data()), sSize*sizeof(s.front()) );
        //        cout<<it.first<<" "<<it.second<<endl;
    }
    osData.close();
}
void ReadSprivMapBinaryFile( const char* comment)
{

    double start = ncnn::get_current_time();
    char path[256];
    sprintf(path, MODEL_DIR "%s.m.dat", comment);
    std::ifstream isData(path, std::ios_base::in | std::ios_base::binary);
    int ssSize;
    isData.read(reinterpret_cast<char *>(&ssSize),sizeof(ssSize));
    if (isData)
    {
        for(int i=0; i<ssSize; i++)
        {
            int key;
            isData.read(reinterpret_cast<char*>(&key), sizeof(key));
            int sSize;
            isData.read(reinterpret_cast<char*>(&sSize), sizeof(sSize));
            std::vector<uint32_t> spirv;
            spirv.resize(sSize);
            isData.read(reinterpret_cast<char*>(spirv.data()), sSize * sizeof(uint32_t));
            ncnn::spriv_map.insert(std::pair<int, std::vector<uint32_t>>(key, spirv));
            //            ncnn::spriv_vectors.push_back(spriv);
            //            printf("%d  %d:{", key, sSize);
            ////            for(int i : spirv){
            ////                printf("%d,", i);
            ////            }
            //            printf("}\n");
        }
    }
    else
    {
        printf("ERROR: Cannot open file 数据.dat");
    }

    double e = ncnn::get_current_time();
    printf("rmap_time %f\n", e-start);
    isData.close();
}


int main(int argc, char** argv)
{

    start_t = ncnn::get_current_time();
    int loop_count = 4;
    int num_threads = ncnn::get_cpu_count();
    int powersave = 0;
    int gpu_device = -1;
    int cooling_down = 1;

    if (argc >= 2)
    {
        loop_count = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        num_threads = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        powersave = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        gpu_device = atoi(argv[4]);
    }
    if (argc >= 6)
    {
        cooling_down = atoi(argv[5]);
    }

#ifdef __EMSCRIPTEN__
    EM_ASM(
        FS.mkdir('/working');
        FS.mount(NODEFS, {root: '.'}, '/working'););
#endif // __EMSCRIPTEN__

    bool use_vulkan_compute = gpu_device != -1;

    g_enable_cooling_down = cooling_down != 0;

    g_loop_count = loop_count;

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

#if NCNN_VULKAN

    double startt_t = ncnn::get_current_time();
    if (use_vulkan_compute)
    {
        g_warmup_loop_count = 10;

        g_vkdev = ncnn::get_gpu_device(gpu_device);

        g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
        g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
    }
    double end_t = ncnn::get_current_time();
    printf("NCNN_VULKAN init   %f\n",end_t -startt_t);
#endif // NCNN_VULKAN

    // default option
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
#if NCNN_VULKAN
    opt.blob_vkallocator = g_blob_vkallocator;
    opt.workspace_vkallocator = g_blob_vkallocator;
    opt.staging_vkallocator = g_staging_vkallocator;
#endif // NCNN_VULKAN
       //    opt.use_winograd_convolution = true;
       //    opt.use_sgemm_convolution = true;
       //    opt.use_int8_inference = true;
       //    opt.use_vulkan_compute = use_vulkan_compute;
       //    opt.use_fp16_packed = true;
       //    opt.use_fp16_storage = true;
       //    opt.use_fp16_arithmetic = true;
       //    opt.use_int8_storage = true;
       //    opt.use_int8_arithmetic = true;
       //    opt.use_packing_layout = true;
       //    opt.use_shader_pack8 = false;
       //    opt.use_image_storage = false;

    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    opt.use_int8_inference = true;
    opt.use_vulkan_compute = use_vulkan_compute;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = true;
    opt.use_int8_arithmetic = true;
    opt.use_packing_layout = true;
    opt.use_shader_pack8 = false;
    opt.use_image_storage = false;

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);
    fprintf(stderr, "cooling_down = %d\n", (int)g_enable_cooling_down);

    char model_name[] = "GoogleNet";
    ncnn::Mat in = ncnn::Mat(227, 227, 3);

    //    char model_name[] = "yolov4-tiny";
    //    ncnn::Mat in = ncnn::Mat(416, 416, 3);

    //    char model_name[] = "mobilenetv2_yolov3";
    //    ncnn::Mat in = ncnn::Mat(300,300,3);

    //    char model_name[] = "mobilenet_yolo";
    //    ncnn::Mat in = ncnn::Mat(416,416,3);
    if (use_vulkan_compute)
    {
        ReadSprivMapBinaryFile(model_name);//create_pipeline() spriv顺序
    }


//    TEST_ = 1 ;
    benchmark(model_name, in, opt);
    //
    //        WriteDataReaderFile(model_name);
    ////       benchmark("AlexNet", ncnn::Mat(227, 227, 3), opt);
    ////       benchmark("GoogleNet", ncnn::Mat(227, 227, 3), opt);
    ////       WriteSprivMapBinaryFile("GoogleNet");
    //       ReadSprivMapBinaryFile("GoogleNet");
    ////       WriteSprivBinaryFile("GoogleNet");
    ////       ReadSprivBinaryFile("GoogleNet");
    //       benchmark("GoogleNet", ncnn::Mat(227, 227, 3), opt);



    //       benchmark("MobileNet", ncnn::Mat(224, 224, 3), opt);
    //       benchmark("MobileNetV2", ncnn::Mat(224, 224, 3), opt);
    //       benchmark("resnet18", ncnn::Mat(224, 224, 3), opt);
    //       benchmark("shufflenet", ncnn::Mat(224, 224, 3), opt);
    //       benchmark("efficientnet_b0", ncnn::Mat(224, 224, 3), opt);
    //       benchmark("resnet50", ncnn::Mat(224, 224, 3), opt);
    //       benchmark("mobilenet_int8", ncnn::Mat(224, 224, 3), opt);
    //       benchmark("googlenet_int8", ncnn::Mat(224, 224, 3), opt);

    // run
    //        benchmark_new("AlexNet", ncnn::Mat(227, 227, 3), opt);
    //        benchmark_new("GoogleNet", ncnn::Mat(227, 227, 3), opt);
    //        benchmark_new("MobileNet", ncnn::Mat(224, 224, 3), opt);
    //        benchmark_new("MobileNet", ncnn::Mat(224, 224, 3), opt);
    //        benchmark_new("MobileNetV2", ncnn::Mat(224, 224, 3), opt);
    //        benchmark_new("resnet18", ncnn::Mat(224, 224, 3), opt);

    //    benchmark_new("resnet18_int8", ncnn::Mat(224, 224, 3), opt);
    //    sleep(50);
    //    sleep(100);

    //    benchmark("squeezenet", ncnn::Mat(227, 227, 3), opt);
    //
    //    benchmark("squeezenet_int8", ncnn::Mat(227, 227, 3), opt);
    //
    //    benchmark("mobilenet", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("mobilenet_int8", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("mobilenet_v2", ncnn::Mat(224, 224, 3), opt);
    //
    //    // benchmark("mobilenet_v2_int8", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("mobilenet_v3", ncnn::Mat(224, 224, 3), opt);
    //
    //        benchmark("shufflenet", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("shufflenet_v2", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("mnasnet", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("proxylessnasnet", ncnn::Mat(224, 224, 3), opt);
    //
    //        benchmark("efficientnet_b0", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("efficientnetv2_b0", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("regnety_400m", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("blazeface", ncnn::Mat(128, 128, 3), opt);
    //
    //    benchmark("googlenet", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("googlenet_int8", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("resnet18", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("resnet18_int8", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("alexnet", ncnn::Mat(227, 227, 3), opt);
    //
    //    benchmark("vgg16", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("vgg16_int8", ncnn::Mat(224, 224, 3), opt);
    //
    //        benchmark("resnet50", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("resnet50_int8", ncnn::Mat(224, 224, 3), opt);
    //
    //    benchmark("squeezenet_ssd", ncnn::Mat(300, 300, 3), opt);
    //
    //    benchmark("squeezenet_ssd_int8", ncnn::Mat(300, 300, 3), opt);
    //
    //    benchmark("mobilenet_ssd", ncnn::Mat(300, 300, 3), opt);
    //
    //    benchmark("mobilenet_ssd_int8", ncnn::Mat(300, 300, 3), opt);
    //
    //    benchmark("mobilenet_yolo", ncnn::Mat(416, 416, 3), opt);
    //
    //    benchmark("mobilenetv2_yolov3", ncnn::Mat(352, 352, 3), opt);
    //
    //    benchmark("yolov4-tiny", ncnn::Mat(416, 416, 3), opt);

#if NCNN_VULKAN
    delete g_blob_vkallocator;
    delete g_staging_vkallocator;
#endif // NCNN_VULKAN

    //    printf("end %f\n", ncnn::get_current_time()-start_t);
    return 0;
}

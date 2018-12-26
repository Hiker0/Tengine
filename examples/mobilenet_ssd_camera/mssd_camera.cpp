/*
 *  V4L2 video capture example
 *
 *  This program can be used and distributed without restrictions.
 *
 *      This program is provided with the V4L2 API
 * see https://linuxtv.org/docs.php for more information
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <iomanip>
#include <vector>
#include <getopt.h>             /* getopt_long() */

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <inttypes.h>

#include <linux/videodev2.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>

#include "tengine_c_api.h"
#include "common.hpp"


#define DEF_PROTO "models/MobileNetSSD_deploy.prototxt"
#define DEF_MODEL "models/MobileNetSSD_deploy.caffemodel"
#define DEF_IMAGE "tests/images/ssd_dog.jpg"

#define CLEAR(x) memset(&(x), 0, sizeof(x))
#define COLS (640)
#define ROWS (480)
#define SSD_IMG_H (300)
#define SSD_IMG_W (300)
using namespace cv;

enum io_method {
        IO_METHOD_READ,
        IO_METHOD_MMAP,
        IO_METHOD_USERPTR,
};

struct buffer {
        void   *start;
        size_t  length;
};

typedef struct buffer* PBUF;

static char            *dev_name;
static enum io_method   io = IO_METHOD_MMAP;
static int              fd = -1;
struct buffer          *buffers;
static unsigned int     n_buffers;
static int              out_buf;
static int              force_format;
static int              frame_count = 70;

static std::string proto_file;
static std::string model_file;
static std::string image_file;
static std::string save_name="save.jpg";
const char *model_name = "mssd_300";
    
cv::Mat yuvImg(ROWS , COLS, CV_8UC2);
cv::Mat rgbImg(ROWS, COLS,CV_8UC3);
cv::Mat resizeImg(SSD_IMG_W, SSD_IMG_H,CV_8UC3);
cv::Mat floatImg(SSD_IMG_W, SSD_IMG_H, CV_32FC3);

static int fpsTick();

struct Box
{
    float x0;
    float y0;
    float x1;
    float y1;
    int class_idx;
    float score;
};

void get_input_data_ssd(std::string& image_file, float* input_data, int img_h,  int img_w)
{
    cv::Mat img = cv::imread(image_file);

    if (img.empty())
    {
        std::cerr << "Failed to read image file " << image_file << ".\n";
        return;
    }
   
    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float *img_data = (float *)img.data;
    int hw = img_h * img_w;

    float mean[3]={127.5,127.5,127.5};
    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = 0.007843* (*img_data - mean[c]);
                img_data++;
            }
        }
    }
}

void post_process_ssd(std::string& image_file,float threshold,float* outdata,int num,std::string& save_name)
{
    std::cout<<"post_process_ssd\n";
    const char* class_names[] = {"background",
                            "aeroplane", "bicycle", "bird", "boat",
                            "bottle", "bus", "car", "cat", "chair",
                            "cow", "diningtable", "dog", "horse",
                            "motorbike", "person", "pottedplant",
                            "sheep", "sofa", "train", "tvmonitor"};

    //cv::Mat img = cv::imread(image_file);
    int raw_h = rgbImg.size().height;
    int raw_w = rgbImg.size().width;
    std::vector<Box> boxes;
    int line_width=raw_w*0.005;
    printf("detect ruesult num: %d \n",num);
    for (int i=0;i<num;i++)
    {
        if(outdata[1]>=threshold)
        {
            Box box;
            box.class_idx=outdata[0];
            box.score=outdata[1];
            box.x0=outdata[2]*raw_w;
            box.y0=outdata[3]*raw_h;
            box.x1=outdata[4]*raw_w;
            box.y1=outdata[5]*raw_h;
            boxes.push_back(box);
            printf("%s\t:%.0f%%\n", class_names[box.class_idx], box.score * 100);
            printf("BOX:( %g , %g ),( %g , %g )\n",box.x0,box.y0,box.x1,box.y1);
        }
        outdata+=6;
    }
    for(int i=0;i<(int)boxes.size();i++)
    {
        Box box=boxes[i];
        cv::rectangle(rgbImg, cv::Rect(box.x0, box.y0,(box.x1-box.x0),(box.y1-box.y0)),cv::Scalar(255, 255, 0),line_width);
        std::ostringstream score_str;
        score_str<<box.score;
        std::string label = std::string(class_names[box.class_idx]) + ": " + score_str.str();
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(rgbImg, cv::Rect(cv::Point(box.x0,box.y0- label_size.height),
                                  cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 0), CV_FILLED);
        cv::putText(rgbImg, label, cv::Point(box.x0, box.y0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    cv::imshow("opencv",rgbImg);
    waitKey(1);
    std::cout<<"======================================\n";
    std::cout<<"[DETECTED IMAGE SAVED]:\t"<< save_name<<"\n";
    std::cout<<"======================================\n";


}

static void errno_exit(const char *s)
{
        fprintf(stderr, "%s error %d, %s\n", s, errno, strerror(errno));
        exit(EXIT_FAILURE);
}

static int xioctl(int fh, int request, void *arg)
{
        int r;

        do {
                r = ioctl(fh, request, arg);
        } while (-1 == r && EINTR == errno);

        return r;
}

static void process_image(const void *p, int size,float *input_data,int img_w,  int img_h)
{
    std::cout<<"process_image\n";
    //int fps = fpsTick();
    
    memcpy(yuvImg.data, p, COLS*ROWS*2);
    cv::cvtColor(yuvImg, rgbImg, CV_YUV2BGR_YUYV);
    
    cv::resize(rgbImg, resizeImg, cv::Size(img_w, img_h));
    resizeImg.convertTo(floatImg, CV_32FC3);
    float *img_data = (float *)floatImg.data;
    int hw = img_h * img_w;

    float mean[3]={127.5,127.5,127.5};
    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = 0.007843* (*img_data - mean[c]);
                img_data++;
            }
        }
    }
    
    //cv::cvtColor(yuvImg, rgbImg, CV_YUV2BGR_YUYV);
    
    //char title[10];
    //sprintf(title, "fps:%d", fps);
    //cv::imshow("opencv",rgbImg);
    //waitKey(1);
    
    /*
    static int frame_number = 0;
    char fn[256];
    sprintf(fn, "%d.raw", frame_number);
    frame_number++;

    uint8_t *pixel = (uint8_t *) rgbImg.data;
    size = COLS*ROWS*3;
    int found = 0;
    for (int i=0; i < size; i++) {
        if (pixel[i] != 0) {
        found = 1;
        break;
        }
    }

    if (found) {
        FILE *f = fopen(fn, "wb");
        if (f == NULL) { printf("Error opening file\n"); exit(EXIT_FAILURE); }
        fwrite(pixel, size, 1, f);
        fclose(f);

        fprintf(stdout, "%s\n", fn);
        fflush(stdout);
    } else {
        fprintf(stdout, "empty image");
    }
    */
    
}

static int read_frame(float *input_data,int img_w,  int img_h)
{
        struct v4l2_buffer buf;
        unsigned int i;
        static uint64_t timestamp;
        uint64_t stamp =0;

        switch (io) {
        case IO_METHOD_READ:
                if (-1 == read(fd, buffers[0].start, buffers[0].length)) {
                        switch (errno) {
                        case EAGAIN:
                                return 0;

                        case EIO:
                                /* Could ignore EIO, see spec. */

                                /* fall through */

                        default:
                                errno_exit("read");
                        }
                }

                process_image(buffers[0].start, buffers[0].length,input_data, img_w, img_h);
                break;

        case IO_METHOD_MMAP:
                CLEAR(buf);

                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory = V4L2_MEMORY_MMAP;

                if (-1 == xioctl(fd, VIDIOC_DQBUF, &buf)) {
                        switch (errno) {
                        case EAGAIN:
                                return 0;

                        case EIO:
                                /* Could ignore EIO, see spec. */

                                /* fall through */

                        default:
                                errno_exit("VIDIOC_DQBUF");
                        }
                }
                stamp = buf.timestamp.tv_sec*1000000+buf.timestamp.tv_usec;
                //printf("timestamp :%ld", timestamp);
                if(timestamp == stamp){
                    break;
                }
                
                assert(buf.index < n_buffers);

                process_image(buffers[buf.index].start, buf.bytesused, input_data, img_w, img_h);

                if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
                        errno_exit("VIDIOC_QBUF");
                break;

        case IO_METHOD_USERPTR:
                CLEAR(buf);

                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory = V4L2_MEMORY_USERPTR;

                if (-1 == xioctl(fd, VIDIOC_DQBUF, &buf)) {
                    
                        switch (errno) {
                        case EAGAIN:
                                return 0;

                        case EIO:
                                /* Could ignore EIO, see spec. */

                                /* fall through */

                        default:
                                errno_exit("VIDIOC_DQBUF");
                        }
                }

                for (i = 0; i < n_buffers; ++i)
                        if (buf.m.userptr == (unsigned long)buffers[i].start
                            && buf.length == buffers[i].length)
                                break;

                assert(i < n_buffers);

                process_image((void *)buf.m.userptr, buf.bytesused, input_data, img_w, img_h);

                if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
                        errno_exit("VIDIOC_QBUF");
                break;
        }

        return 1;
}

static void mainloop(void)
{
        unsigned int count;

        count = frame_count;
        

            // init tengine
        init_tengine_library();
        if (request_tengine_version("0.1") < 0)
            return ;
        if (load_model(model_name, "caffe", proto_file.c_str(), model_file.c_str()) < 0)
            return ;
        std::cout << "load model done!\n";
    
        // create graph
        graph_t graph = create_runtime_graph("graph", model_name, NULL);
        if (!check_graph_valid(graph))
        {
            std::cout << "create graph0 failed\n";
            return ;
        }

        
        int repeat_count = 1;
        const char *repeat = std::getenv("REPEAT_COUNT");

        if (repeat)
            repeat_count = std::strtoul(repeat, NULL, 10);
        
        int node_idx=0;
        int tensor_idx=0;
        tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
        if(!check_tensor_valid(input_tensor))
        {
            printf("Get input node failed : node_idx: %d, tensor_idx: %d\n",node_idx,tensor_idx);
            return;
        }

        
        // input
        int img_h = 300;
        int img_w = 300;
        int img_size = img_h * img_w * 3;
        float *input_data = (float *)malloc(sizeof(float) * img_size);
        int dims[] = {1, 3, img_h, img_w};
        set_tensor_shape(input_tensor, dims, 4);
        
        prerun_graph(graph);
    
        while (1) {
           printf("Reading frame\n");
                for (;;) {
                        fd_set fds;
                        struct timeval tv;
                        int r;

                        FD_ZERO(&fds);
                        FD_SET(fd, &fds);

                        /* Timeout. */
                        tv.tv_sec = 2;
                        tv.tv_usec = 0;

                        r = select(fd + 1, &fds, NULL, NULL, &tv);

                        if (-1 == r) {
                                if (EINTR == errno)
                                        continue;
                                errno_exit("select");
                        }

                        if (0 == r) {
                                fprintf(stderr, "select timeout\n");
                                exit(EXIT_FAILURE);
                        }
                        
                        
                        if (read_frame(input_data,img_w, img_h))
                        {
                            std::cout<<"run_graph\n";
                            /* EAGAIN - continue select loop. */
                            set_tensor_buffer(input_tensor, input_data, img_size * 4);
                            run_graph(graph, 1);

                            //gettimeofday(&t1, NULL);
                            //float mytime = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
                            //total_time += mytime;

                            //std::cout << "--------------------------------------\n";
                            //std::cout << "repeat " << repeat_count << " times, avg time per run is " << total_time / repeat_count << " ms\n";
                            
                            tensor_t out_tensor = get_graph_output_tensor(graph, 0,0);//"detection_out");
                            int out_dim[4];
                            get_tensor_shape( out_tensor, out_dim, 4);

                            float *outdata = (float *)get_tensor_buffer(out_tensor);
                            int num=out_dim[1];
                            float show_threshold=0.5;
                            
                            post_process_ssd(image_file,show_threshold, outdata, num,save_name);
                            put_graph_tensor(out_tensor);
                            
                            
                            std::cout<<"run end\n";
                            
                            break;
                        }

                }
                
        }
        
        postrun_graph(graph);
        
        free(input_data);
        
        put_graph_tensor(input_tensor);
        
        

        destroy_runtime_graph(graph);
        remove_model(model_name);

        return;
}

static void stop_capturing(void)
{
        enum v4l2_buf_type type;

        switch (io) {
        case IO_METHOD_READ:
                /* Nothing to do. */
                break;

        case IO_METHOD_MMAP:
        case IO_METHOD_USERPTR:
                type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                if (-1 == xioctl(fd, VIDIOC_STREAMOFF, &type))
                        errno_exit("VIDIOC_STREAMOFF");
                break;
        }
}

static void start_capturing(void)
{
        unsigned int i;
        enum v4l2_buf_type type;

        switch (io) {
        case IO_METHOD_READ:
                /* Nothing to do. */
                break;

        case IO_METHOD_MMAP:
                for (i = 0; i < n_buffers; ++i) {
                        struct v4l2_buffer buf;

                        CLEAR(buf);
                        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                        buf.memory = V4L2_MEMORY_MMAP;
                        buf.index = i;

                        if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
                                errno_exit("VIDIOC_QBUF");
                }
                type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                if (-1 == xioctl(fd, VIDIOC_STREAMON, &type))
                        errno_exit("VIDIOC_STREAMON");
                break;

        case IO_METHOD_USERPTR:
                for (i = 0; i < n_buffers; ++i) {
                        struct v4l2_buffer buf;

                        CLEAR(buf);
                        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                        buf.memory = V4L2_MEMORY_USERPTR;
                        buf.index = i;
                        buf.m.userptr = (unsigned long)buffers[i].start;
                        buf.length = buffers[i].length;

                        if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
                                errno_exit("VIDIOC_QBUF");
                }
                type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                if (-1 == xioctl(fd, VIDIOC_STREAMON, &type))
                        errno_exit("VIDIOC_STREAMON");
                break;
        }
}

static void uninit_device(void)
{
        unsigned int i;

        switch (io) {
        case IO_METHOD_READ:
                free(buffers[0].start);
                break;

        case IO_METHOD_MMAP:
                for (i = 0; i < n_buffers; ++i)
                        if (-1 == munmap(buffers[i].start, buffers[i].length))
                                errno_exit("munmap");
                break;

        case IO_METHOD_USERPTR:
                for (i = 0; i < n_buffers; ++i)
                        free(buffers[i].start);
                break;
        }

        free(buffers);
}

static void init_read(unsigned int buffer_size)
{
        buffers = (PBUF)calloc(1, sizeof(*buffers));

        if (!buffers) {
                fprintf(stderr, "Out of memory\n");
                exit(EXIT_FAILURE);
        }

        buffers[0].length = buffer_size;
        buffers[0].start = malloc(buffer_size);

        if (!buffers[0].start) {
                fprintf(stderr, "Out of memory\n");
                exit(EXIT_FAILURE);
        }
}

static void init_mmap(void)
{
        struct v4l2_requestbuffers req;

        CLEAR(req);

        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
                if (EINVAL == errno) {
                        fprintf(stderr, "%s does not support "
                                 "memory mapping\n", dev_name);
                        exit(EXIT_FAILURE);
                } else {
                        errno_exit("VIDIOC_REQBUFS");
                }
        }

        if (req.count < 2) {
                fprintf(stderr, "Insufficient buffer memory on %s\n",
                         dev_name);
                exit(EXIT_FAILURE);
        }

        buffers = (PBUF)calloc(req.count, sizeof(*buffers));

        if (!buffers) {
                fprintf(stderr, "Out of memory\n");
                exit(EXIT_FAILURE);
        }

        for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
                struct v4l2_buffer buf;

                CLEAR(buf);

                buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory      = V4L2_MEMORY_MMAP;
                buf.index       = n_buffers;

                if (-1 == xioctl(fd, VIDIOC_QUERYBUF, &buf))
                        errno_exit("VIDIOC_QUERYBUF");

                buffers[n_buffers].length = buf.length;
                buffers[n_buffers].start =
                        mmap(NULL /* start anywhere */,
                              buf.length,
                              PROT_READ | PROT_WRITE /* required */,
                              MAP_SHARED /* recommended */,
                              fd, buf.m.offset);

                if (MAP_FAILED == buffers[n_buffers].start)
                        errno_exit("mmap");
        }
}

static void init_userp(unsigned int buffer_size)
{
        struct v4l2_requestbuffers req;

        CLEAR(req);

        req.count  = 4;
        req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_USERPTR;

        if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
                if (EINVAL == errno) {
                        fprintf(stderr, "%s does not support "
                                 "user pointer i/o\n", dev_name);
                        exit(EXIT_FAILURE);
                } else {
                        errno_exit("VIDIOC_REQBUFS");
                }
        }

        buffers = (PBUF)calloc(4, sizeof(*buffers));

        if (!buffers) {
                fprintf(stderr, "Out of memory\n");
                exit(EXIT_FAILURE);
        }

        for (n_buffers = 0; n_buffers < 4; ++n_buffers) {
                buffers[n_buffers].length = buffer_size;
                buffers[n_buffers].start = malloc(buffer_size);

                if (!buffers[n_buffers].start) {
                        fprintf(stderr, "Out of memory\n");
                        exit(EXIT_FAILURE);
                }
        }
}

static void init_device(void)
{
        struct v4l2_capability cap;
        struct v4l2_cropcap cropcap;
        struct v4l2_crop crop;
        struct v4l2_format fmt;
        unsigned int min;

        if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap)) {
                if (EINVAL == errno) {
                        fprintf(stderr, "%s is no V4L2 device\n",
                                 dev_name);
                        exit(EXIT_FAILURE);
                } else {
                        errno_exit("VIDIOC_QUERYCAP");
                }
        }

        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
                fprintf(stderr, "%s is no video capture device\n",
                         dev_name);
                exit(EXIT_FAILURE);
        }

        switch (io) {
        case IO_METHOD_READ:
                if (!(cap.capabilities & V4L2_CAP_READWRITE)) {
                        fprintf(stderr, "%s does not support read i/o\n",
                                 dev_name);
                        exit(EXIT_FAILURE);
                }
                break;

        case IO_METHOD_MMAP:
        case IO_METHOD_USERPTR:
                if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
                        fprintf(stderr, "%s does not support streaming i/o\n",
                                 dev_name);
                        exit(EXIT_FAILURE);
                }
                break;
        }


        /* Select video input, video standard and tune here. */

  struct v4l2_format format = {0};
  format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  format.fmt.pix.width = COLS;
  format.fmt.pix.height = ROWS;
  format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  format.fmt.pix.field = V4L2_FIELD_NONE;
  int retval = xioctl(fd, VIDIOC_S_FMT, &format);
  if (retval == -1) { perror("Setting format\n"); return; }


//
//        CLEAR(cropcap);
//
//        cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//
//        if (0 == xioctl(fd, VIDIOC_CROPCAP, &cropcap)) {
//                crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//                crop.c = cropcap.defrect; /* reset to default */
//
//                if (-1 == xioctl(fd, VIDIOC_S_CROP, &crop)) {
//                        switch (errno) {
//                        case EINVAL:
//                                /* Cropping not supported. */
//                                break;
//                        default:
//                                /* Errors ignored. */
//                                break;
//                        }
//                }
//        } else {
//                /* Errors ignored. */
//        }
//
//
//        CLEAR(fmt);
//
//        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//        if (force_format) {
//                fmt.fmt.pix.width       = 640;
//                fmt.fmt.pix.height      = 480;
//                fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
//                fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;
//
//                if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt))
//                        errno_exit("VIDIOC_S_FMT");
//
//                /* Note VIDIOC_S_FMT may change width and height. */
//        } else {
//                /* Preserve original settings as set by v4l2-ctl for example */
//                if (-1 == xioctl(fd, VIDIOC_G_FMT, &fmt))
//                        errno_exit("VIDIOC_G_FMT");
//        }
//
//        /* Buggy driver paranoia. */
//        min = fmt.fmt.pix.width * 2;
//        if (fmt.fmt.pix.bytesperline < min)
//                fmt.fmt.pix.bytesperline = min;
//        min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
//        if (fmt.fmt.pix.sizeimage < min)
//                fmt.fmt.pix.sizeimage = min;
//
        switch (io) {
        case IO_METHOD_READ:
                init_read(fmt.fmt.pix.sizeimage);
                break;

        case IO_METHOD_MMAP:
                init_mmap();
                break;

        case IO_METHOD_USERPTR:
                init_userp(fmt.fmt.pix.sizeimage);
                break;
        }
}

static void close_device(void)
{
        if (-1 == close(fd))
                errno_exit("close");

        fd = -1;
}

static void open_device(void)
{
        struct stat st;

        if (-1 == stat(dev_name, &st)) {
                fprintf(stderr, "Cannot identify '%s': %d, %s\n",
                         dev_name, errno, strerror(errno));
                exit(EXIT_FAILURE);
        }

        if (!S_ISCHR(st.st_mode)) {
                fprintf(stderr, "%s is no device\n", dev_name);
                exit(EXIT_FAILURE);
        }

        fd = open(dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

        if (-1 == fd) {
                fprintf(stderr, "Cannot open '%s': %d, %s\n",
                         dev_name, errno, strerror(errno));
                exit(EXIT_FAILURE);
        }
}


static int fpsTick()
{

    static clock_t last=clock();
    static clock_t avgDuration = 0;
    static float alpha = 1.f/10.f;
    static int frameCount = 0;
    
    clock_t now = clock();
    clock_t delta = now-last;
    
    printf("delta clock:%d\n", delta);
    last = now;
    
    
    frameCount++;
    
    int fps = 0;
    if(1 == frameCount)
    {
        avgDuration = delta;
    }
    else
    {
        avgDuration = avgDuration * (1.f - alpha) + delta * alpha;
    }
    
    fps = (1.f * CLOCKS_PER_SEC/ avgDuration);
    printf("fps :%d\n", fps);
    
    
}

static void usage(FILE *fp, int argc, char **argv)
{
        fprintf(fp,
                 "Usage: %s [options]\n\n"
                 "Version 1.3\n"
                 "Options:\n"
                 "-d | --device name   Video device name [%s]\n"
                 "-h | --help          Print this message\n"
                 "-m | --mmap          Use memory mapped buffers [default]\n"
                 "-r | --read          Use read() calls\n"
                 "-u | --userp         Use application allocated buffers\n"
                 "-o | --output        Outputs stream to stdout\n"
                 "-f | --format        Force format to 640x480 YUYV\n"
                 "-c | --count         Number of frames to grab [%i]\n"
                 "",
                 argv[0], dev_name, frame_count);
}

static const char short_options[] = "d:hmruofc:";

static const struct option
long_options[] = {
        { "device", required_argument, NULL, 'd' },
        { "help",   no_argument,       NULL, 'h' },
        { "mmap",   no_argument,       NULL, 'm' },
        { "read",   no_argument,       NULL, 'r' },
        { "userp",  no_argument,       NULL, 'u' },
        { "output", no_argument,       NULL, 'o' },
        { "format", no_argument,       NULL, 'f' },
        { "count",  required_argument, NULL, 'c' },
        { 0, 0, 0, 0 }
};

int main(int argc, char **argv)
{
        dev_name = "/dev/video0";

        for (;;) {
                int idx;
                int c;

                c = getopt_long(argc, argv,
                                short_options, long_options, &idx);

                if (-1 == c)
                        break;

                switch (c) {
                case 0: /* getopt_long() flag */
                        break;

                case 'd':
                        dev_name = optarg;
                        break;

                case 'h':
                        usage(stdout, argc, argv);
                        exit(EXIT_SUCCESS);

                case 'm':
                        io = IO_METHOD_MMAP;
                        break;

                case 'r':
                        io = IO_METHOD_READ;
                        break;

                case 'u':
                        io = IO_METHOD_USERPTR;
                        break;

                case 'o':
                        out_buf++;
                        break;

                case 'f':
                        force_format++;
                        break;

                case 'c':
                        errno = 0;
                        frame_count = strtol(optarg, NULL, 0);
                        if (errno)
                                errno_exit(optarg);
                        break;

                default:
                        usage(stderr, argc, argv);
                        exit(EXIT_FAILURE);
                }
        }
        
    
        //cvNamedWindow("opencv", CV_WINDOW_AUTOSIZE);
        
        const std::string root_path = get_root_path();
        std::string save_name="save.jpg";
        proto_file = root_path + DEF_PROTO;
        model_file = root_path + DEF_MODEL;
        image_file = root_path + DEF_IMAGE;
    
        open_device();
        init_device();
        start_capturing();
        mainloop();
        stop_capturing();
        uninit_device();
        close_device();
        fprintf(stderr, "\n");
        waitKey(0);
        return 0;
}
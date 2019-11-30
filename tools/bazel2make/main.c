/* ---------------------------------------------------
 * [How to build TFLite GPU Delegate V2 for AARCH64]
 * ---------------------------------------------------
 *
 * $ git clone https://github.com/tensorflow/tensorflow.git
 * $ cd tensorflow
 * $ git checkout r2.1
 * $ bazel build -s -c opt --copt="-DMESA_EGL_NO_X11_HEADERS" tensorflow/lite/delegates/gpu:delegate &> bazel_log.txt
 * $ bazel2log bazel_log.txt
 *    --> add the contens of "out.txt" to "tensorflow/lite/tools/make/Makefile"
 * $ make -j 4  -f ./tensorflow/lite/tools/make/Makefile BUILD_WITH_NNAPI=false TARGET=aarch64
 * $ scp tensorflow/lite/tools/make/gen/aarch64_armv8-a/lib/libtensorflow-lite.a hoge@192.168.1.1:/home/hoge
 */

#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
    FILE    *fp;
    FILE    *fpdst;
    char    strReadBuf[1024];

    if (argc < 2)
    {
        fprintf (stderr, "usage: bazel2make bazel_log_file\n");
        return 0;
    }
    
    fp = fopen (argv[1], "r" );
    if (fp == NULL)
    {
        fprintf( stderr, "FATAL ERROR at \"%s\"(%d)\n", __FILE__, __LINE__ );
        return -1;
    }

    fpdst = fopen ("out.txt", "w");
    if (fpdst == NULL)
    {
        fprintf( stderr, "FATAL ERROR at \"%s\"(%d)\n", __FILE__, __LINE__ );
        return -1;
    }
    
#if 0
    [pattern-1]
    SUBCOMMAND: # @flatbuffers//:flatc_library [action 'Compiling external/flatbuffers/src/util.cpp [for host]']
        ==> ignore the last "[for host]']"

    [pattern-2]
    SUBCOMMAND: # //tensorflow/lite/delegates/gpu/gl/kernels:depthwise_conv [action 'Compiling tensorflow/lite/delegates/gpu/gl/kernels/depthwise_conv.cc']
        ==> ignore the last "']"
#endif

    while ( fgets( strReadBuf, 1024, fp ) != NULL )
    {
        /* extract lines starting with "SUBCOMMAND" */
        if (strncmp (strReadBuf, "SUBCOMMAND", 10) == 0)
        {
            /* extract lines including "Compiling" */
            char *lptmp = strstr (strReadBuf, "Compiling ");
            if (lptmp)
            {
                lptmp += 10; /* skip "Compiling" */
#if 0
                int len = strlen(lptmp);
                lptmp[len-4] = '\0';

                /* ignore the last bracket */
                len = strlen(lptmp);
                if (lptmp[len-1] == ']')
                {
                    lptmp[len-11] = '\0';
                }
#else
                /* ignore the last single-quotation mark */
                char *p = strrchr (lptmp, '\'');
                *p = '\0';

                /* ignore the last bracket */
                int len = strlen(lptmp);
                if (lptmp[len-1] == ']')
                {
                    p = strrchr (lptmp, '[');
                    *p = '\0';
                }
#endif
                /* Emit file name for make */
                fprintf (fpdst, "CORE_CC_ALL_SRCS += %s\n", lptmp);
            }
            else
                fprintf (fpdst, "## %s", strReadBuf);
        }
	}

    fclose (fp);
    fclose (fpdst);

    return 0;
}

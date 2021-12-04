#include "cuda.h"
#include "math.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <GL/freeglut.h>
#include "../common/book.h"
#include "../common/cpu_anim.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_ELEVATION 2.3f
#define MIN_ELEVATION 1.0f
#define SPEED   0.25f
#define FIRE_PROB   0.0005f
#define WIND  2
#define DIR  "E"

// these exist on the GPU side
texture<float>  texConstSrc;
texture<float>  texIn;
texture<float>  texOut;

//creates circles
bool checkCircle(int x, int y, int xCenter, int yCenter, int radius){
    if(sqrt(pow(x-xCenter,2)+pow(y-yCenter,2)) < radius){
        return true;
    }
    return false;
}

// struct for fire attributes
struct pixelData {
    bool on_fire;
    bool cooling;
    float temp;
};

// updates probability based on elevation
__device__ float update_prob(int cTile, int nTile, bool dstOut){
    float tile_elevation;
    float prob; 

    if (dstOut)
        tile_elevation = tex1Dfetch(texIn,nTile);
    else
        tile_elevation = tex1Dfetch(texOut,nTile);

    if (tile_elevation < tex1Dfetch(texIn,cTile))
        prob = FIRE_PROB * 1.5f;
    else if (tile_elevation > tex1Dfetch(texIn,cTile))
        prob = FIRE_PROB / 2.5f;
    else
        prob = FIRE_PROB;

    if (prob > 1.0) 
        prob = 1.0;
    
    return prob;
}

// init tiles array
__device__ void init_tile_arr(int arr[], int first, int second, int third, int fourth){
    arr[0] = first;
    arr[1] = second;
    arr[2] = third;
    arr[3] = fourth;
}

//Main kernel for Fire simulation
__global__ void fire_sim_kernel(float *dst, bool dstOut, pixelData *pixels, curandState *curandstate) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;    

    int left = offset - 1;
    int right = offset + 1;
    if (x == 0)   left++;
    if (x == DIM-1) right--; 

    int top = offset - DIM;
    int bottom = offset + DIM;
    if (y == 0)   top += DIM;
    if (y == DIM-1) bottom -= DIM;

    float prob; 
    if (!pixels[offset].on_fire and !pixels[offset].cooling){
        // do when current position is not on fire or burnt
        int tiles[4];
        float tile_order = curand_uniform(curandstate+x);

        // shuffle tiles order (weird behavior accord when not shuffled)
        // This is an optimized solution to the problem and leads to about 15 additional 
        // ms to each animation frame
        if (tile_order <= 0.25)
            init_tile_arr(tiles, top, bottom, right, left);
        else if (tile_order <= 0.5)
            init_tile_arr(tiles, bottom, right, left, top);
        else if (tile_order <= 0.75)
            init_tile_arr(tiles, right, left, top, bottom);
        else if (tile_order <= 1)
            init_tile_arr(tiles, left, top, bottom, right);

        // for tiles (top, left, right bottom) check if on fire and if spread to current tile
        for (int tile:tiles){
            prob = update_prob(offset, tile, dstOut);
            if (pixels[tile].on_fire && curand_uniform(curandstate+x) >=  1 - prob){
                pixels[offset].on_fire = true;
                break;
            }
        }

        // check for addition spread due to wind
        for (int i = 0; i <= WIND; i++){
            if (DIR == "N" && y > i && pixels[offset - (DIM * (i+1))].on_fire){
                prob = update_prob(offset, offset - (DIM * (i+1)), dstOut);
                if (curand_uniform(curandstate+x) >= 1 - prob ){
                    pixels[offset].on_fire = true;
                    break;
                }
            }
            if (DIR == "E" && x > i && pixels[offset - (i+1)].on_fire){
                prob = update_prob(offset, offset - (i+1), dstOut);
                if (curand_uniform(curandstate+x) >= 1 - prob ){
                    pixels[offset].on_fire = true;
                    break;
                }
            }
            if (DIR == "W" && x <= i && pixels[offset + (i+1)].on_fire){
                prob = update_prob(offset, offset + (i+1), dstOut);
                if (curand_uniform(curandstate+x) >= 1 - prob ){
                    pixels[offset].on_fire = true;
                    break;
                }
            }
            if (DIR == "S" && y <= i && pixels[offset + (DIM * (i+1))].on_fire){
                prob = update_prob(offset, offset + (DIM * (i+1)), dstOut);
                if (curand_uniform(curandstate+x) >= 1 - prob ){
                    pixels[offset].on_fire = true;
                    break;
                }
            }
        }
    }

    // process fire life for current tile
   if (pixels[offset].on_fire){
        if (pixels[offset].cooling){
            if (pixels[offset].temp <= 60)
                pixels[offset].on_fire = false;
            else
                pixels[offset].temp -= 0.25;
        }
        else if (pixels[offset].temp == 1000){
            pixels[offset].temp -= 0.25;
            pixels[offset].cooling = true;
        }
        else if (pixels[offset].temp < 1000)
            pixels[offset].temp += 0.25;
    }
}

// used for map generation from assignment 4 heat.cu
__global__ void blend_kernel( float *dst,
                              bool dstOut ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset - 1;
    int right = offset + 1;
    if (x == 0)   left++;
    if (x == DIM-1) right--; 

    int top = offset - DIM;
    int bottom = offset + DIM;
    if (y == 0)   top += DIM;
    if (y == DIM-1) bottom -= DIM;

    float   t, l, c, r, b;
    if (dstOut) {
        t = tex1Dfetch(texIn,top);
        l = tex1Dfetch(texIn,left);
        c = tex1Dfetch(texIn,offset);
        r = tex1Dfetch(texIn,right);
        b = tex1Dfetch(texIn,bottom);

    } else {
        t = tex1Dfetch(texOut,top);
        l = tex1Dfetch(texOut,left);
        c = tex1Dfetch(texOut,offset);
        r = tex1Dfetch(texOut,right);
        b = tex1Dfetch(texOut,bottom);
    }
    dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

// used for map generation from assignment 4 heat.cu
__global__ void copy_const_kernel( float *iptr ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = tex1Dfetch(texConstSrc,offset);
    if (c != MIN_ELEVATION)
        iptr[offset] = c;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *output_bitmap;
    float           *dev_inSrc;
    float           *dev_outSrc;
    float           *dev_constSrc;
    CPUAnimBitmap  *bitmap;
    pixelData       *fire_map;

    cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;
};

// used to convert float values to colors for fire simulation
// darker greens equal higher elevation
// red to yellow for fire
// yellow for higher temperatures
__global__ void float_to_map( unsigned char *optr,
                              const float *outSrc,
                              pixelData *p ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float l = outSrc[offset];

    if (p[offset].on_fire) {
        optr[offset*4 + 0] = 255;
        optr[offset*4 + 1] = 255 * (p[offset].temp / 1000);
        optr[offset*4 + 2] = 0;
        optr[offset*4 + 3] = 255;
    }
    else if (p[offset].cooling and p[offset].temp <= 60){
        optr[offset*4 + 0] = 0;
        optr[offset*4 + 1] = 0;
        optr[offset*4 + 2] = 0;
        optr[offset*4 + 3] = 255;
    }else{
        optr[offset*4 + 0] = 0;
        optr[offset*4 + 1] = 200/l;
        optr[offset*4 + 2] = 0;
        optr[offset*4 + 3] = 255;
    }
}

// used to convert float values to colors for map generation (shades of green)
// darker greens equal higher elevation
__global__ void float_to_map( unsigned char *optr,
                              const float *outSrc ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float l = outSrc[offset];

    optr[offset*4 + 0] = 0;
    optr[offset*4 + 1] = 200/l;
    optr[offset*4 + 2] = 0;
    optr[offset*4 + 3] = 255;
}

// initilize curand
__global__ void setup_kernel(curandState *state){
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(clock64(), idx, 0, &state[idx]);
}

// GLU animation loop for fire simulation modifed from heat.cu
void anim_fire( DataBlock *d, int ticks ) {
    HANDLE_ERROR( cudaEventRecord( d->start, 0 ) );
    dim3    blocks(DIM/8,DIM/8);
    dim3    threads(8,8);
    CPUAnimBitmap  *bitmap = d->bitmap;
    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState));
    if (d->totalTime == 0)
        setup_kernel<<<1,1>>>(d_state);

    // since tex is global and bound, we have to use a flag to
    // select which is in/out per iteration
    volatile bool dstOut = true;
    for (int i=0; i<90; i++) {
        float   *in, *out;
        if (dstOut) {
            in  = d->dev_inSrc;
            out = d->dev_outSrc;
        } else {
            out = d->dev_inSrc;
            in  = d->dev_outSrc;
        }
        copy_const_kernel<<<blocks,threads>>>( in );
        fire_sim_kernel<<<blocks,threads>>>( out, dstOut, d->fire_map, d_state );
        dstOut = !dstOut;
    }
    float_to_map<<<blocks,threads>>>( d->output_bitmap,
                                        d->dev_inSrc, d->fire_map );

    HANDLE_ERROR( cudaMemcpy( bitmap->get_ptr(),
                              d->output_bitmap,
                              bitmap->image_size(),
                              cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR( cudaEventRecord( d->stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( d->stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        d->start, d->stop ) );
    d->totalTime += elapsedTime;
    ++d->frames;
    printf( "Average Time per frame:  %3.1f ms\n",
            d->totalTime/d->frames  );
    if (d->totalTime >= 10000.0)
        glutLeaveMainLoop();
}

// GLU animation loop for map gen modifed from heat.cu
void anim_gpu( DataBlock *d, int ticks ) {
    HANDLE_ERROR( cudaEventRecord( d->start, 0 ) );
    dim3    blocks(DIM/16,DIM/16);
    dim3    threads(16,16);
    
    CPUAnimBitmap  *bitmap = d->bitmap;
    float* output_map = (float*)malloc( bitmap->image_size() );;

    // since tex is global and bound, we have to use a flag to
    // select which is in/out per iteration
    volatile bool dstOut = true;
    for (int i=0; i<90; i++) {
        float   *in, *out;
        if (dstOut) {
            in  = d->dev_inSrc;
            out = d->dev_outSrc;
        } else {
            out = d->dev_inSrc;
            in  = d->dev_outSrc;
        }
        copy_const_kernel<<<blocks,threads>>>( in );
        blend_kernel<<<blocks,threads>>>( out, dstOut );
        dstOut = !dstOut;
    }
    float_to_map<<<blocks,threads>>>( d->output_bitmap,
                                        d->dev_inSrc);

    HANDLE_ERROR( cudaMemcpy( bitmap->get_ptr(),
                              d->output_bitmap,
                              bitmap->image_size(),
                              cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR( cudaEventRecord( d->stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( d->stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        d->start, d->stop ) );
    d->totalTime += elapsedTime;
    ++d->frames;
    printf( "Average Time per frame:  %3.1f ms\n",
            d->totalTime/d->frames  );
    if (d->totalTime >= 7000.0) {
        if (dstOut) {
            HANDLE_ERROR( cudaMemcpy( output_map,
                              d->dev_outSrc,
                              bitmap->image_size(),
                              cudaMemcpyDeviceToHost ) );
        } else {
            HANDLE_ERROR( cudaMemcpy( output_map,
                              d->dev_inSrc,
                              bitmap->image_size(),
                              cudaMemcpyDeviceToHost ) );
        }

        FILE *fp;
        fp = fopen("./test.dat", "wb+");
         for(int i = 0; i<bitmap->image_size() ; i++){
            float f =  output_map[i];
            fwrite(&f, sizeof(float), 1, fp);
        }
        fclose(fp);
        glutLeaveMainLoop();
    }
}

// clean up memory allocated on the GPU
void anim_exit( DataBlock *d ) {
    cudaUnbindTexture( texIn );
    cudaUnbindTexture( texOut );
    cudaUnbindTexture( texConstSrc );
    HANDLE_ERROR( cudaFree( d->dev_inSrc ) );
    HANDLE_ERROR( cudaFree( d->fire_map ) );
    HANDLE_ERROR( cudaFree( d->dev_outSrc ) );
    HANDLE_ERROR( cudaFree( d->dev_constSrc ) );

    HANDLE_ERROR( cudaEventDestroy( d->start ) );
    HANDLE_ERROR( cudaEventDestroy( d->stop ) );
}

// init elevation map and fire map
void init_host(DataBlock data, int imageSize, bool gen_map){
    std::vector<int> xCords;
    std::vector<int> yCords;
    int x,y;
    pixelData *tiles;

    const int HILLS = 3;
    const int RADIUS = 20;
    tiles = (pixelData*)malloc(sizeof(pixelData) * imageSize);
    int xfire = rand() % DIM;
    int yfire = rand() % DIM;

    for (int i=0; i < HILLS; i++) {
        x = rand() % DIM;
        y = rand() % DIM;
        xCords.push_back(x);
        yCords.push_back(y);
     }

    float *elevation = (float*)malloc( imageSize );

     for (int i=0; i<DIM*DIM; i++) {
        int x = i % DIM;
        int y = i / DIM;
        tiles[i].on_fire = false;
        tiles[i].cooling = false;
        tiles[i].temp = 0;
        if (checkCircle(x,y,xfire,yfire,10)){
            tiles[i].on_fire = true;
        }
        
        if ( gen_map ) {
            elevation[i] = MIN_ELEVATION;
            for (int j = 0; j < HILLS; j++) {
                if (checkCircle(x,y,xCords[j],yCords[j],RADIUS)) {
                    elevation[i] = MAX_ELEVATION;
                    break;
                }
            }
        }
     }

    HANDLE_ERROR( cudaMemcpy(data.fire_map, tiles, sizeof(pixelData)*imageSize, cudaMemcpyHostToDevice));

    if (gen_map){
        HANDLE_ERROR( cudaMemcpy( data.dev_constSrc, elevation,
                                imageSize,
                                cudaMemcpyHostToDevice ) );    

        HANDLE_ERROR( cudaMemcpy( data.dev_inSrc, elevation,
                                imageSize,
                                cudaMemcpyHostToDevice ) );  
    }

    free( elevation );

}

int main( int argc, char *argv[] ) {
    if (argc > 1) { 
        srand(time(NULL));
        DataBlock   data;
        
        CPUAnimBitmap bitmap( DIM, DIM, &data );
        data.bitmap = &bitmap;
        data.totalTime = 0;
        data.frames = 0;
        HANDLE_ERROR( cudaEventCreate( &data.start ) );
        HANDLE_ERROR( cudaEventCreate( &data.stop ) );

        int imageSize = bitmap.image_size();

        HANDLE_ERROR( cudaMalloc( (void**)&data.output_bitmap,
                                imageSize ) );

        // assume float == 4 chars in size (ie rgba)
        HANDLE_ERROR( cudaMalloc( (void**)&data.dev_inSrc,
                                imageSize ) );
        HANDLE_ERROR( cudaMalloc( (void**)&data.dev_outSrc,
                                imageSize ) );
        HANDLE_ERROR( cudaMalloc( (void**)&data.dev_constSrc,
                                imageSize ) );
        HANDLE_ERROR( cudaMalloc( (void**)&data.fire_map,
                                sizeof(pixelData)* imageSize ) );

        HANDLE_ERROR( cudaBindTexture( NULL, texConstSrc,
                                    data.dev_constSrc,
                                    imageSize ) );

        HANDLE_ERROR( cudaBindTexture( NULL, texIn,
                                    data.dev_inSrc,
                                    imageSize ) );

        HANDLE_ERROR( cudaBindTexture( NULL, texOut,
                                    data.dev_outSrc,
                                    imageSize ) );

                     
        if (strcmp(argv[1], "--gen_map") == 0 || strcmp(argv[1], "-g") == 0){
            init_host(data, imageSize, true);
            bitmap.anim_and_exit( (void (*)(void*,int))anim_gpu, (void (*)(void*))anim_exit );
        }
        else if (strcmp(argv[1], "--sim") == 0 || strcmp(argv[1], "-s") == 0){
            FILE *fp;
            float *elevation = (float*)malloc( imageSize );
            fp = fopen("./test.dat", "rb");
            for(int i = 0; i< DIM*DIM; i++){
                float f;
                fread(&f, sizeof(float), 1, fp);
                elevation[i] = f;
            }
            fclose(fp);

            init_host(data, imageSize, false);
            HANDLE_ERROR( cudaMemcpy( data.dev_constSrc, elevation,
                              imageSize,
                              cudaMemcpyHostToDevice ) );    

            HANDLE_ERROR( cudaMemcpy( data.dev_inSrc, elevation,
                                    imageSize,
                                    cudaMemcpyHostToDevice ) );
            free( elevation );

            bitmap.anim_and_exit( (void (*)(void*,int))anim_fire, (void (*)(void*))anim_exit );
        }
    }
    else{
        printf("Use arg --sim     -s for fire simulation\n");
        printf("Use arg --gen_map -g for map generation\n\n");
        printf("You must run map generation or have a map saved at ./test.dat\n");
    }
}

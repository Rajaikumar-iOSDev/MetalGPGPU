//
//  main.m
//  MetalGPGPU
//
//  Created by Rajai Kumar on 07/07/24.
//

#import "MetalAdder.h"
#import <mach/mach_time.h>
// The number of floats in each array, and the size of the arrays in bytes.
const unsigned int arrayLength = 1 << 24;
const unsigned int bufferSize = arrayLength * sizeof(float);

@implementation MetalAdder
{
    id<MTLDevice> _mDevice;

    // The compute pipeline generated from the compute kernel in the .metal shader file.
    id<MTLComputePipelineState> _mComputeValuesPSO;

    // The command queue used to pass commands to the device.
    id<MTLCommandQueue> _mCommandQueue;

    // Buffers to hold data.
    id<MTLBuffer> _mBufferA;
    id<MTLBuffer> _mBufferB;
    id<MTLBuffer> _mBufferResult;

}

- (instancetype) initWithDevice: (id<MTLDevice>) device
{
    self = [super init];
    if (self)
    {
        _mDevice = device;

        NSError* error = nil;

        // Load the shader files with a .metal file extension in the project

        id<MTLLibrary> defaultLibrary = [_mDevice newDefaultLibrary];
        if (defaultLibrary == nil)
        {
            NSLog(@"Failed to find the default library.");
            return nil;
        }

        id<MTLFunction> computeValues = [defaultLibrary newFunctionWithName:@"compute_Values"];
        if (computeValues == nil)
        {
            NSLog(@"Failed to find the adder function.");
            return nil;
        }

        // Create a compute pipeline state object.
        _mComputeValuesPSO = [_mDevice newComputePipelineStateWithFunction: computeValues error:&error];
        if (_mComputeValuesPSO == nil)
        {
            //  If the Metal API validation is enabled, you can find out more information about what
            //  went wrong.  (Metal API validation is enabled by default when a debug build is run
            //  from Xcode)
            NSLog(@"Failed to created pipeline state object, error %@.", error);
            return nil;
        }

        _mCommandQueue = [_mDevice newCommandQueue];
        if (_mCommandQueue == nil)
        {
            NSLog(@"Failed to find the command queue.");
            return nil;
        }
    }

    return self;
}

- (void) prepareData
{
    // Allocate three buffers to hold our initial data and the result.
    _mBufferA = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    _mBufferB = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    _mBufferResult = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];

    [self generateRandomFloatData:_mBufferA];
    [self generateRandomFloatData:_mBufferB];
}

- (void) sendComputeCommand
{  uint64_t start = mach_absolute_time();
    // Create a command buffer to hold commands.
    id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
    assert(commandBuffer != nil);

    // Start a compute pass.
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    assert(computeEncoder != nil);

    [self encodeAddCommand:computeEncoder];

    // End the compute pass.
    [computeEncoder endEncoding];
    
    // Execute the command.
    [commandBuffer commit];

    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    [commandBuffer waitUntilCompleted];
     uint64_t end = mach_absolute_time();
    uint64_t elapsed = end - start;

    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    double elapsedNano = (double)elapsed * (double)info.numer / (double)info.denom;


    printf("Time taken - GPU: %f nanoseconds\n", elapsedNano);
    [self usingDispatchQueue];
}

- (void)encodeAddCommand:(id<MTLComputeCommandEncoder>)computeEncoder {

    // Encode the pipeline state object and its parameters.
    [computeEncoder setComputePipelineState:_mComputeValuesPSO];
    [computeEncoder setBuffer:_mBufferA offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufferB offset:0 atIndex:1];
    [computeEncoder setBuffer:_mBufferResult offset:0 atIndex:2];

    MTLSize gridSize = MTLSizeMake(arrayLength, 1, 1);

    // Calculate a threadgroup size.
    NSUInteger threadGroupSize = _mComputeValuesPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > arrayLength)
    {
        threadGroupSize = arrayLength;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    // Encode the compute command.
    [computeEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

- (void) generateRandomFloatData: (id<MTLBuffer>) buffer
{
    float* dataPtr = buffer.contents;

    for (unsigned long index = 0; index < arrayLength; index++)
    {
        dataPtr[index] = (float)rand()/(float)(RAND_MAX);
    }
}

- (void) usingDispatchQueue
{
    float* a = _mBufferA.contents;
    float* b = _mBufferB.contents;
    //float* result = _mBufferResult.contents; use this to verify results
    uint64_t start = mach_absolute_time();
dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    dispatch_apply(arrayLength, queue, ^(size_t index) {
        // Compute the expected dot product
        float dotProduct = a[index] * b[index];

        // Apply the sigmoid function to the dot product
        float expected = 1.0 / (1.0 + exp(-dotProduct));
        //printf("Expected: %f \n", expected);
        // // To Verify the results
        // if (fabs(result[index] - expected) > 1e-5) // Allowing a small tolerance for floating-point comparison
        // {
        //     printf("Compute ERROR: index=%zu result=%g vs %g=expected\n",
        //            index, result[index], expected);
        //     assert(fabs(result[index] - expected) <= 1e-5);
        // }
    });

    uint64_t end = mach_absolute_time();
    uint64_t elapsed = end - start;

    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    double elapsedNano = (double)elapsed * (double)info.numer / (double)info.denom;

    printf("Time taken - DispatchQueue: %f nanoseconds\n", elapsedNano);
    
}
@end

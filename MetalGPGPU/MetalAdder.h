//
//  main.m
//  MetalGPGPU
//
//  Created by Rajai Kumar on 07/07/24.
//


#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

@interface MetalAdder : NSObject
- (instancetype) initWithDevice: (id<MTLDevice>) device;
- (void) prepareData;
- (void) sendComputeCommand;
@end

NS_ASSUME_NONNULL_END

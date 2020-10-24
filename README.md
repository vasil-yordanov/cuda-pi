# compile

```
nvcc pi.cu -o gpu_pi.exe
nvcc pi.c -o cpu_pi.exe
```

# run
```
gpu_pi.exe
cpu_pi.exe
```

# Example run on GPU
```
 CUDADevice with properties:

                      Name: 'GeForce GTX 1050 Ti'
                     Index: 1
         ComputeCapability: '6.1'
            SupportsDouble: 1
             DriverVersion: 10.1000
            ToolkitVersion: 10
        MaxThreadsPerBlock: 1024
          MaxShmemPerBlock: 49152
        MaxThreadBlockSize: [1024 1024 64]
               MaxGridSize: [2.1475e+09 65535 65535]
                 SIMDWidth: 32
               TotalMemory: 4.2950e+09
           AvailableMemory: 3.4361e+09
       MultiprocessorCount: 6
              ClockRateKHz: 1620000
               ComputeMode: 'Default'
      GPUOverlapsTransfers: 1
    KernelExecutionTimeout: 1
          CanMapHostMemory: 1
           DeviceSupported: 1
            DeviceSelected: 1
```


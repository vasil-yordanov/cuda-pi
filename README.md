# compile

```
nvcc gpu_pi.cu -o gpu_pi.exe
nvcc cpu_pi.cu -o cpu_pi.exe
```

# run
```
gpu_pi.exe
cpu_pi.exe
```

# System information
```
System Manufacturer	Acer
System Model	Predator PH317-52
System Type	x64-based PC
OS Name	Microsoft Windows 10 Pro
Processor	Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz, 2208 Mhz, 6 Core(s), 12 Logical Processor(s)
Installed Physical Memory (RAM)	32.0 GB
Name	NVIDIA GeForce GTX 1050 Ti
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
# Summary comparison
GPU v.s 1 core of CPU
```
CPU =    25,056,689 points/sec
GPU = 7,242,720,735 points/sec

GPU/CPU = 289 times faster
```

# Example CPU Run - 1 Core
```
C:\>cpu_pi.exe
   time (ms)  |  total points   |  points in 1/4 circle |       estimated pi        |          error
------------------------------------------------------------------------------------------------------------
             0          65536000                51467519            3.14132806396484       -0.00026458962495
          2889         131072000               102935849            3.14135281372070       -0.00023983986909
          5461         196608000               154408701            3.14145306396484       -0.00013958962495
          8061         262144000               205878665            3.14145912170410       -0.00013353188569
         10668         327680000               257348042            3.14145559082031       -0.00013706276948
         13201         393216000               308817425            3.14145329793294       -0.00013935565685
         15797         458752000               360288281            3.14146450369699       -0.00012814989280
         18416         524288000               411757317            3.14145902252197       -0.00013363106782
         20953         589824000               463228302            3.14146797688802       -0.00012467670177
         23562         655360000               514705233            3.14151143188477       -0.00008122170502
         26134         720896000               566181092            3.14154103781960       -0.00005161577019
         28928         786432000               617651750            3.14153925577799       -0.00005339781180
         31353         851968000               669121825            3.14153501070463       -0.00005764288516
         33915         917504000               720591779            3.14153084455218       -0.00006180903761
         36354         983040000               772059650            3.14151875813802       -0.00007389545177
         38917        1048576000               823533362            3.14153046417236       -0.00006218941743
         42161        1114112000               875000219            3.14151618149701       -0.00007647209278
         44984        1179648000               926468012            3.14150665961372       -0.00008599397607
         47711        1245184000               977940139            3.14151206247430       -0.00008059111549
         50402        1310720000              1029409949            3.14150985412598       -0.00008279946381
         53340        1376256000              1080882183            3.14151490129743       -0.00007775229236
         56015        1441792000              1132350370            3.14150826194070       -0.00008439164909
         58810        1507328000              1183819730            3.14150531271230       -0.00008734087749
         61536        1572864000              1235290437            3.14150603485107       -0.00008661873872
         64166        1638400000              1286759061            3.14150161376953       -0.00009103982026
         66779        1703936000              1338234211            3.14151285259540       -0.00007980099439
         69511        1769472000              1389711605            3.14152833161531       -0.00006432197448
         72077        1835008000              1441183349            3.14153038896833       -0.00006226462146
         74583        1900544000              1492653772            3.14152952417834       -0.00006312941145
         77112        1966080000              1544122400            3.14152506510417       -0.00006758848562
         79606        2031616000              1595590864            3.14152057081653       -0.00007208277326
         82035        2097152000              1647066204            3.14152947235107       -0.00006318123872
         84565        2162688000              1698530799            3.14151796098189       -0.00007469260790
         87205        2228224000              1750000602            3.14151647590188       -0.00007617768791
         89703        2293760000              1801469883            3.14151416538783       -0.00007848820196
         92272        2359296000              1852943348            3.14151907687717       -0.00007357671262
         94789        2424832000              1904414711            3.14152025542388       -0.00007239816591
         97292        2490368000              1955883319            3.14151694689299       -0.00007570669680
         99975        2555904000              2007350546            3.14151164675982       -0.00008100682997
        102754        2621440000              2058820487            3.14151075286865       -0.00008190072114
        105509        2686976000              2110289438            3.14150842880621       -0.00008422478358
        107975        2752512000              2161761374            3.14151055326916       -0.00008210032063
        110987        2818048000              2213239135            3.14152084705441       -0.00007180653538
        113596        2883584000              2264715231            3.14152836331454       -0.00006429027525
```

# Example GPU Run
```
C:\>gpu_pi.exe
   time (ms)  |  total points   |  points in 1/4 circle |       estimated pi        |          error
------------------------------------------------------------------------------------------------------------
          2262       16384000000             12867813613            3.14155605786133       -0.00003659572846
          4501       32768000000             25735814558            3.14157892553711       -0.00001372805268
          6740       49152000000             38603858735            3.14159006632487       -0.00000258726492
          8979       65536000000             51471826569            3.14159097711182       -0.00000167647797
         11218       81920000000             64339808299            3.14159220209961       -0.00000045149018
         13466       98304000000             77207673723            3.14158828625488       -0.00000436733491
         15720      114688000000             90075676644            3.14159028473772       -0.00000236885207
         17974      131072000000            102943729723            3.14159331430054        0.00000066071075
         20228      147456000000            115811674071            3.14159272111003        0.00000006752024
         22482      163840000000            128679604901            3.14159191652832       -0.00000073706147
         24736      180224000000            141547490551            3.14159025548207       -0.00000239810772
         26990      196608000000            154415453351            3.14159044089762       -0.00000221269217
         29243      212992000000            167283512887            3.14159241449444       -0.00000023909535
         31497      229376000000            180151508541            3.14159299213518        0.00000033854539
         33750      245760000000            193019435101            3.14159236818034       -0.00000028540945
         36003      262144000000            205887338751            3.14159147264099       -0.00000118094880
         38262      278528000000            218755297810            3.14159147819968       -0.00000117539011
         40532      294912000000            231623229693            3.14159111454264       -0.00000153904715
         42802      311296000000            244491321449            3.14159284345446        0.00000018986467
         45071      327680000000            257359230555            3.14159216986084       -0.00000048372895
         47340      344064000000            270227227558            3.14159258228702       -0.00000007130277
         49610      360448000000            283095226596            3.14159297980291        0.00000032621312
         51879      376832000000            295963232654            3.14159341726817        0.00000076367838
         54149      393216000000            308831207848            3.14159350431315        0.00000085072336
         56418      409600000000            321699211009            3.14159385750977        0.00000120391998
         58687      425984000000            334567179062            3.14159385387245        0.00000120028266
         60956      442368000000            347435094825            3.14159337768555        0.00000072409576
         63225      458752000000            360303108143            3.14159378612409        0.00000113253430
         65493      475136000000            373171034747            3.14159343638032        0.00000078279053
         67763      491520000000            386038984124            3.14159329527995        0.00000064169016
         70031      507904000000            398907007913            3.14159374931483        0.00000109572504
         72300      524288000000            411774902106            3.14159318623352        0.00000053264373
         74569      540672000000            424642863785            3.14159315655333        0.00000050296354
         76838      557056000000            437510780180            3.14159280345244        0.00000014986265
         79106      573440000000            450378815131            3.14159329750977        0.00000064391998
         81375      589824000000            463246788970            3.14159334967719        0.00000069608740
         83644      606208000000            476114747687            3.14159329924382        0.00000064565403
         85913      622592000000            488982756627            3.14159357413523        0.00000092054544
         88183      638976000000            501850711982            3.14159349948668        0.00000084589689
         90452      655360000000            514718645370            3.14159329449463        0.00000064090484
         92721      671744000000            527586691391            3.14159377019222        0.00000111660243
         94990      688128000000            540454742382            3.14159425212751        0.00000159853772
         97258      704512000000            553322687755            3.14159411198106        0.00000145839127
         99527      720896000000            566190565198            3.14159360128507        0.00000094769528
        101796      737280000000            579058502764            3.14159343947483        0.00000078588504
```

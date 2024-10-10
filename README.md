# VIF_Benchmark
#### ✨News:
[2024-07-16] 우리의 논문 《[DRMF: Degradation-Robust Multi-Modal Image Fusion via Composable Diffusion Prior]([https://www.sciencedirect.com/science/article/pii/S1566253523001860](https://openreview.net/forum?id=BwXrlBweab))》가 《ACM MM 2024》에 정식으로 채택되었습니다! [[논문 다운로드](https://openreview.net/pdf?id=BwXrlBweab)] [[코드](https://github.com/Linfeng-Tang/DRMF)]
## Star History
<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="
      https://api.star-history.com/svg?repos=Linfeng-Tang/VIF-Benchmark&type=Date&theme=dark
    "
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="
      https://api.star-history.com/svg?repos=Linfeng-Tang/VIF-Benchmark&type=Date
    "
  />
  <img
    alt="Star History Chart"
    src="https://api.star-history.com/svg?repos=Linfeng-Tang/VIF-Benchmark&type=Date"
  />
</picture>

## 한국어 버전

> 딥러닝 기반 적외선 및 가시광선 이미지 융합 방법을 모두 통합한 프레임워크를 제작했습니다.
> 벤치마크는 다음과 같은 방법을 포함합니다:

1. [CSF](https://github.com/hanna-xu/CSF)
2. [CUFD](https://github.com/Meiqi-Gong/CUFD)
3. [DIDFuse](https://github.com/Meiqi-Gong/CUFD)
4. [DIVFusion](https://github.com/Linfeng-Tang/DIVFusion)
5. [DenseFuse](https://github.com/hli1221/imagefusion_densefuse)
6. [FusionGAN](https://github.com/jiayi-ma/FusionGAN)
7. [GAN-FM](https://github.com/yuanjiteng/GAN-FM)
8. [GANMcC](https://github.com/jiayi-ma/GANMcC)
9. [IFCNN](https://github.com/uzeful/IFCNN)
10. [NestFuse](https://github.com/hli1221/imagefusion-nestfuse)
11. [PIAFusion](https://github.com/Linfeng-Tang/PIAFusion)
12. [PMGI](https://github.com/HaoZhang1018/PMGI_AAAI2020)
13. [RFN-Nest](https://github.com/hli1221/imagefusion-rfn-nest)
14. [SDNet](https://github.com/HaoZhang1018/SDNet)
15. [STDFusionNet](https://github.com/Linfeng-Tang/STDFusionNet)
16. [SeAFusion](https://github.com/Linfeng-Tang/SeAFusion)
17. [SuperFusion](https://github.com/Linfeng-Tang/SuperFusion)
18. [SwinFusion](https://github.com/Linfeng-Tang/SwinFusion)
19. [TarDAL](https://github.com/JinyuanLiu-CV/TarDAL)
20. [U2Fusion](https://github.com/hanna-xu/U2Fusion)
21. [UMF-CMGR](https://github.com/wdhudiekou/UMF-CMGR)

**사용법:**
  
  1) 적외선 이미지는 `./datasets/test_imgs/ir` 폴더에, 가시광선 이미지는 `./datasets/test_imgs/vi` 폴더에 저장
  
  2) **python All_in_One.py**를 실행하면, 융합 결과는 **./Results**에 저장됩니다.

  3) 만약 모든 방법을 실행할 필요가 없다면, **All_in_One.py** 파일의 **Method_list**를 수정

  4) Docker나 Conda를 사용하여 Tensorflow v1 및 Pytorch를 함께 사용할 수 있도록 환경을 설정했습니다.

```bash
## 1. Image 생성
docker build -t tf0 -f Dockerfile .
### 1-1. 만약 다양한 환경을 설정하고 싶다면, 아래와 같이 설정하세요.
docker build -t tf1:20.12 --build-arg UID=$UID --build-arg USER_NAME=$USER --build-arg CUDA_VERSION=11.3.1 --build-arg CUDA=11.3 --build-arg CUDNN=$USER --build-arg PYTHON_VERSION=3.6 --build-arg CONDA_ENV_NAME=timer -f DockerfileV2 .
## 2. Container 실행
docker run -d -it --rm --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia -p 8888:8888 -v $PWD:/workspace/VIF tf1 /bin/bash
## 3. Container 접속
docker exec -it tf1 /bin/bash
```


> 정리하기 쉽지 않으니, 우리 프로젝트에 **Star**를 달아주시고 아래의 문헌을 인용해주세요. 여러분의 지원은 지속적인 업데이트에 큰 힘이 됩니다.

```BibTeX
@inproceedings{Tang2024DRMF,
    title={DRMF: Degradation-Robust Multi-Modal Image Fusion via Composable Diffusion Prior},
    author={Tang, Linfeng and Deng, Yuxin and Yi, Xunpeng and Yan, Qinglong and Yuan, Yixuan and Ma, Jiayi},
    booktitle=Proceedings of the ACM International Conference on Multimedia,
    year={2024}
}
@article{TangSeAFusion,
    title = {Image fusion in the loop of high-level vision tasks: A semantic-aware real-time infrared and visible image fusion network},
    author = {Linfeng Tang and Jiteng Yuan and Jiayi Ma},
    journal = {Information Fusion},
    volume = {82},
    pages = {28-42},
    year = {2022},
    issn = {1566-2535},
    publisher={Elsevier}
}
@article{TangSeAFusion,
    title = {Image fusion in the loop of high-level vision tasks: A semantic-aware real-time infrared and visible image fusion network},
    author = {Linfeng Tang and Jiteng Yuan and Jiayi Ma},
    journal = {Information Fusion},
    volume = {82},
    pages = {28-42},
    year = {2022},
    issn = {1566-2535},
    publisher={Elsevier}
}
@article{Tang2022SuperFusion,
    title={SuperFusion: A versatile image registration and fusion network with semantic awareness},
    author={Tang, Linfeng and Deng, Yuxin and Ma, Yong and Huang, Jun and Ma, Jiayi},
    journal={IEEE/CAA Journal of Automatica Sinica},
    volume={9},
    number={12},
    pages={2121--2137},
    year={2022},
    publisher={IEEE}
}
@article{Tang2022DIVFusion,
    title={DIVFusion: Darkness-free infrared and visible image fusion},
    author={Tang, Linfeng and Xiang, Xinyu and Zhang, Hao and Gong, Meiqi and Ma, Jiayi},
    journal={Information Fusion},
    volume = {91},
    pages = {477-493},
    year = {2023},
    publisher={Elsevier}
}
@article{Tang2022PIAFusion,
    title={PIAFusion: A progressive infrared and visible image fusion network based on illumination aware},
    author={Tang, Linfeng and Yuan, Jiteng and Zhang, Hao and Jiang, Xingyu and Ma, Jiayi},
    journal={Information Fusion},
    volume = {83-84},
    pages = {79-92},
    year = {2022},
    issn = {1566-2535},
    publisher={Elsevier}
}
@article{Ma2021STDFusionNet,
    title={STDFusionNet: An Infrared and Visible Image Fusion Network Based on Salient Target Detection},
    author={Jiayi Ma, Linfeng Tang, Meilong Xu, Hao Zhang, and Guobao Xiao},
    journal={IEEE Transactions on Instrumentation and Measurement},
    year={2021},
    volume={70},
    number={},
    pages={1-13},
    doi={10.1109/TIM.2021.3075747}，
    publisher={IEEE}
}
@article{Tang2022Survey,
    title={Deep learning-based image fusion: A survey},
    author={Tang, Linfeng and Zhang, Hao and Xu, Han and Ma, Jiayi},  
    journal={Journal of Image and Graphics}
    volume={28},
    number={1},
    pages={3--36},
    year={2023}
}
```
> Github의 제한으로 인해 현재 DIVFusion의 체크포인트에는 **decom.ckpt** 파일이 없습니다. 해당 체크포인트는 저자의 원본 프로젝트 [DIVFusion](https://github.com/Linfeng-Tang/DIVFusion)에서 다운로드할 수 있으며, 저에게 연락해도 다운로드가 가능합니다.

> 원본 프로젝트에 대한 문제는 해당 프로젝트의 저자에게 문의해 주세요. 이 프로젝트와 관련된 질문이 있다면 **linfeng0419@gmail.com** 또는 **QQ: 2458707789**(이름 + 학교를 명시하여 연락)로 연락해 주세요. 프로젝트 이슈는 알림이 가지 않으므로, 즉각적인 응답이 어려운 점 양해 부탁드립니다.

> 일부 융합 결과는 아래와 같습니다:
> ![TNO 데이터셋의 17.png](https://github.com/Linfeng-Tang/VIF_Benchmark/blob/main/Demo/17.png)
> 
> ![TNO 데이터셋의 21.png](https://github.com/Linfeng-Tang/VIF_Benchmark/blob/main/Demo/21.png)
> 
> ![RoadScene 데이터셋의 175.png](https://github.com/Linfeng-Tang/VIF_Benchmark/blob/main/Demo/175.png)
> 
> ![MSRS 데이터셋의 00633D.png](https://github.com/Linfeng-Tang/VIF_Benchmark/blob/main/Demo/00633D.png)
> 
> ![MSRS 데이터셋의 01023N.png](https://github.com/Linfeng-Tang/VIF_Benchmark/blob/main/Demo/01023N.png)
>

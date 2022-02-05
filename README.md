# kotrocr : 
한국어 필기체 이미지 인식 모델을 Transformer 기반 encoder-decoder 구조를 이용해 학습을 진행한다.
- 모델 : [영어 필기체 인식 모델](https://arxiv.org/abs/2109.10282)을 참고하여 한국어에 맞게 구조를 수정한다.
- 데이터셋 : AI hub 한글 글자체 이미지, text generator 생성 이미지, 수동 제작 필기 이미지

<br>

## Dataset
 **인쇄체**와 **필기체**를 이용하였고, 추가로 **text generator**를 이용해 직접 데이터셋을 제작해 학습하였다.

### textgenerator 사용법
- trdg/fonts/ko/에 원하는 ttf font 파일을 넣는다.
- backgrounds에 원하는 배경을 넣는다.
- 이미지 크기를 선택한다.
생성한다.
```
import subprocess, sys, tqdm, random
count = 500 #~7000
total_count = 10000
left = []
for i in tqdm.tqdm(range(0,int(total_count/count))):
    fd_size = 64
    text_color = '#020202'
    p = subprocess.run(
        args=[sys.executable, './trdg/run.py', '--output_dir', f'hyukhun/word_test/', '-hyukhun', '-f',
         str(fd_size), '-c', str(count), '--start_count', str(count*i), '-fd', 'trdg/fonts/test', 
         '-id',f'backgrounds/test/', '-b', '3', '-tc', f'{text_color}', '-na', '2', '-t', '4'], capture_output=True, encoding='utf-8')
    if p.stderr:
        print(p.stderr)
    print(p.stdout)
    if 'too many function definitions' in p.stderr:
        left.append(p.stdout.split('__split__')[1].strip('[]'))
    print(f'{i}_done')
    print('-'*80)

```






## 데이터 폴더 설명
- 이미지와 라벨은 따로 올리지 않았으니 참고 바란다.
- 데이터를 다음과 같이 놓으면 된다.
```

  |--data              
    |--images              # 글씨 이미지
    |--labels.txt          # 정답 셋

```
<br>


I. 라벨링 파일 구조는 아래와 같다.<br>
```
# labels.txt
# {filename}.jpg {label}
0.jpg 여기가
1.jpg 라벨이
2.jpg 있는
3.jpg 곳입니다.
...



## multi_train으로 학습
```
python multi_train.py --gpus 3 --mode {finetune or embed} --devices 0,1,2
```

## eval로 결과치 평가

## test로 여러 테스트 데이터 예측한다.


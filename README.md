# trocr_trainer

this is for training ko_trocr


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


## multi_train으로 학습
```
python multi_train.py --gpus 3 --mode {finetune or embed} --devices 0,1,2
```

## eval로 결과치 평가

## test로 여러 테스트 데이터 예측한다.

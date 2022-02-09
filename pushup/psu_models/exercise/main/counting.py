import pandas as pd

# 운동종류 추론을 위해 input image를 8개로 맞춤
def divide_exer(csv_file):
    """
        exercise model에 input image를 8개 맞춰주기 위해
        4분위로 나눠서 각 범위에 2개씩 random으로 추출
    """
    df = pd.read_csv(csv_file)
    ars_y = df['right_shoulder_y'].astype('float32')
    divide_ars_y = pd.cut(ars_y, 4, labels=['1','2','3','4']) 

    divide_list = []
    divide_list.extend(list(divide_ars_y[divide_ars_y=='1'].sample(n=2, random_state=1).index)) 
    divide_list.extend(list(divide_ars_y[divide_ars_y=='2'].sample(n=2, random_state=1).index))
    divide_list.extend(list(divide_ars_y[divide_ars_y=='3'].sample(n=2, random_state=1).index))
    divide_list.extend(list(divide_ars_y[divide_ars_y=='4'].sample(n=2, random_state=1).index))
    return divide_list

# 푸시업 횟수 카운트 (횟수만)
def split_count(csv_file):
    """
        df: 24개의 keypoint와 이미지 파일 명 데이터프레임
        24개의 관절 중 가장 정확성이 높은 right_shoulder_y를 기준으로 푸시업 개수를 카운트
        1. right_shoulder_y 값을 1과 0으로 나눔
        2. right_shoulder_y의 이상치 해결: 1(혹은 0)이 10프레임 이상 반복됐을 때 정상치로 해석, 10프레임 미만 반복 됐을 때 그 다음 값으로 변경
        3. 1,0 의 반복을 푸시업 한번 으로 카운트
    """
    df = pd.read_csv(csv_file)
    ars_y = df['right_shoulder_y'].astype('float32') # 오른쪽 어깨 좌표
    mean_ars = sum(ars_y) / len(ars_y)  
    
    bin_ars = ars_y.copy()
    bin_ars.loc[bin_ars>mean_ars] = 1 # 평균값보다 크면 1
    bin_ars.loc[bin_ars!=1] = 0 # 작으면 0

    i = 0
    start = i
    end = i
    now = bin_ars[start]
    mode = [now]
    while(True):
        if i == len(bin_ars):
            break
        if now != bin_ars[i]:
            if (end - start + 1) < 10:
                bin_ars[start:end+1] = bin_ars[i] 
                now = bin_ars[i]
                end = i
            else:
                now = bin_ars[i]
                start = i
                end = i
                i += 1
        else:
            end = i
            i += 1

    a = pd.DataFrame({"image":df['image'], "ars_y": ars_y, "bin_ars":bin_ars})
    a['bin_ars'] = a['bin_ars'].astype(int)

    count_idx=[] # 횟수별 프레임의 idx를 담는 리스트
    start = 0 # 횟수당 start idx
    now = a['bin_ars'][start]
    i = 0
    mode = [now]
    while(True):
        if i == len(a['bin_ars']):
            if len(mode)==2:
                count_idx.append([idx for idx in range(start,i)])
            break
        if now != a['bin_ars'][i]:
            if len(mode)== 2:
                #0,1 한번 사이클 돌았을 때:
                count_idx.append([idx for idx in range(start,i)])
                now = a['bin_ars'][i]
                mode = [now] # mode 리스트 초기화
                start = i
            else:
                now = a['bin_ars'][i]
                mode.append(now)
                i+=1
        else:
            i+=1

    return count_idx # 횟수별 프레임의 idx를 담는 리스트
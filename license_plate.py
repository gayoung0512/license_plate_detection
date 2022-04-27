import cv2
import numpy as np
import pytesseract
from PIL import Image
#필요한 라이브러리

img = cv2.imread('license_plate_2.jpg')
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

height, width, channel = img.shape

#이미지 전처리
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #grayscale
cv2.imwrite('gray.jpg', gray)
cv2.imshow("gray", gray)
cv2.waitKey(0)

blur = cv2.GaussianBlur(gray, (3, 3), 0) #가우시안 블러 (원본 이미지, 필터 크기, 표준 편차)
cv2.imwrite('blur.jpg', blur)
cv2.imshow("blur", blur)
cv2.waitKey(0)

canny = cv2.Canny(blur, 100, 200)#canny 함수
cv2.imwrite('canny.jpg', canny)
cv2.imshow("canny", canny)
cv2.waitKey(0)

#윤곽선 그리기
contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#상하구조 구성하지 않고 모든 contour 찾기
contour_result = np.zeros((height, width, channel), dtype=np.uint8)
#-1 주는 것: 전체 contour 다 찾기
cv2.drawContours(contour_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
contour_result= np.zeros((height, width, channel), dtype=np.uint8)
#모든 값이 0인 배열 생성

contours_dict = []
#윤곽선의 정보 저장

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)#contour의 사각형 범위 구하기 ( x,y 좌표 높이 너비 저장)
    cv2.rectangle(contour_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)
    # 사각형 그려보기
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),#사각형의 중심 좌표 저장
        'cy': y + (h / 2) #사각형의 중심 좌표 저장
    })

cv2.imwrite('contour_result.jpg', contour_result)
cv2.imshow("contour_result", contour_result)
cv2.waitKey(0)

#번호판의 크기 대략 가정
#번호판 글자처럼 생긴 애들만 남기기

MIN_WIDTH, MIN_HEIGHT = 2, 8# 최소 너비와 높이
MIN_RATIO, MAX_RATIO = 0.45, 1.0# 가로대비 세로 비율의 최소 최대값

MIN_AREA = 200 #최소 넓이
possible_contours = [] #가능한 값을 다시 저장

cnt = 0
for x in contours_dict:#for문 통해
    area = x['w'] * x['h']#넓이 계산
    ratio = x['w'] / x['h']#비율 계산

#조건들에 맞게 비교
    if area > MIN_AREA \
            and x['w'] > MIN_WIDTH and x['h'] > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
        x['idx'] = cnt
        cnt += 1
        possible_contours.append(x)#다시 저장 idx도 같이 저장

# visualize possible contours
contour_result2 = np.zeros((height, width, channel), dtype=np.uint8)

for x in possible_contours:
    cv2.rectangle(contour_result2, pt1=(x['x'], x['y']), pt2=(x['x'] + x['w'], x['y'] + x['h']), color=(255, 255, 255),
                  thickness=2)

cv2.imwrite('contour_result2.jpg', contour_result2)
cv2.imshow("contour_result2", contour_result2)
cv2.waitKey(0)


#숫자들의 배열 모양 정하기
#진짜 번호판 추려내기

MAX_DIAG_MULTIPLYER = 5  # 5 contour 중심 사이의 길이 (대각선의 다섯배)
MAX_ANGLE_DIFF = 12.0  # 12.0
MAX_AREA_DIFF = 0.5  # 0.5
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 3


def find_chars(contour_list): #재귀 함수
    matched_result_idx = []#최종적으로 남는 index 값들

    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:#같은 contour은 비교 하지 않아도 됨
                continue

            dx = abs(d1['cx'] - d2['cx'])#중심점 사이의 거리
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
            #첫번째 contour의 대각선 길이

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            #벡터 사이의 거리 계산

            #각도 계산하기

            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx)) # arctan 함수 사용
                #degree 함수 이용해 도로 바꾼다.

            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            #기준에 맞게 넘어줌

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                    and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                    and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        # append this contour
        matched_contours_idx.append(d1['idx'])

#후보군의 개수가 3보다 작으면 제외 한국의 번호판 7자리
        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

#최종 후보군에 들지 않은 contour
        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                #matched_contour_idx 아닌 contour 배열에 넣기
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
#index가 같은 값만 추출
        #재귀 함수로 돌리기
        # recursive
        recursive_contour_list = find_chars(unmatched_contour)

        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx


result_idx = find_chars(possible_contours)

matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

# visualize possible contours
contour_result3 = np.zeros((height, width, channel), dtype=np.uint8)


for r in matched_result:
    for d in r:
        #         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(contour_result3, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=2)

cv2.imwrite('contour_result3.jpg', contour_result3)
cv2.imshow("contour_result3", contour_result3)
cv2.waitKey(0)




#번호판 수평 회전

PLATE_WIDTH_PADDING = 1.3 #가로 패딩 값
PLATE_HEIGHT_PADDING = 1.5#세로 패딩 값
MIN_PLATE_RATIO = 3 #최소 비율
MAX_PLATE_RATIO = 10 #최대 비율

#배열 생성
plate_imgs = [] #번호판 이미지 저장
plate_infos = []#번호판 정보 저장

for i, matched_chars in enumerate(matched_result):
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])
    #x 방향에 대해 순차적으로 정렬

    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2# x 중심 좌표 구하기
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2 #y 중심 좌표 구하기

    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

     #번호판 간격 삼각형 기준으로 세타각 구하기
    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy'] # 삼각형 높이 구하기
    triangle_hypotenus = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )

    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
    #라디안 값 각도로 바꾸기
    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
    #수평 맞춰서 이미지 회전시키기

    img_rotated = cv2.warpAffine(img, M=rotation_matrix, dsize=(width, height))

    #이미지 크롭하기
    img_cropped = cv2.getRectSubPix(
        img_rotated,
        patchSize=(int(plate_width), int(plate_height)),
        center=(int(plate_cx), int(plate_cy))
    )

#번호판 비율 맞지 않을 경우
    if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[
        0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
        continue


    plate_imgs.append(img_cropped)

    plate_infos.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })

cv2.imwrite('img_cropped.jpg', img_cropped)
cv2.imshow("img_cropped",img_cropped)
cv2.waitKey(0)


#이미지 크롭한 거 다시 전처리하기
img_cropped = cv2.bilateralFilter(img_cropped, 11, 17, 17)
cv2.imshow("license_plate", img_cropped)

cv2.waitKey(0)

ret, thresh = cv2.threshold(img_cropped,80,255,cv2.THRESH_BINARY)
cv2.imshow("binary", thresh)
cv2.waitKey(0)


text = pytesseract.image_to_string(thresh, lang='kor', config=r'-c preserve_interword_spaces=1 --psm 6 --oem 3 -l kor_lstm+eng+osd --tessdata-dir "C:\Program Files (x86)\Tesseract-OCR\tessdata"')
#문자 인식, 경로 설정
#한글 인식에 취약

print("License Plate: ", text)
cv2.waitKey(0)

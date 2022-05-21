# license_plate_detection


## 차량 번호판 인식 알고리즘
![car_license_plate1][pic1]



## 이미지 흑백 처리
 - 컬러 이미지를 그레이스케일로 변환(Grayscaling)
 -  RGB 평균값 계산

## Canny Edge Detection을 이용한 윤곽선 검출

![car_license_plate2][pic2]

1) 가우시안 필터링 이용한 노이즈 제거

![car_license_plate3][pic3]
2) Sobel Kernel 사용해 Gradient 크기/편미분 벡터 구하기

![car_license_plate4][pic4]
3) 비최대 억제(non-maximum supression)

![car_license_plate5][pic5]
4) 히스테리시스 에지 트래킹(hysteresis edge tracking)


## Contour 검출 및 contour 배열 정렬

- 각각의 contour 사이 기울기 차이와 간격이 일정 범위 내인 경우 count -> 간격이 가장 좁고 기울기 차이가 가장 적은 contour 값 탐색
- 번호판 시작점으로 설정 후 bubble 정렬 이용해 contour 배열 정리

## 번호판 문자 인식

- Tesseract OCR 사용

# 차량 번호판 인식

![car_license_plate6][pic6]
- 이미지 전처리

![car_license_plate7][pic7]
- Edge가 추출된 이미지에서 Contour 찾아내기

![car_license_plate7][pic8]
- 최종 번호판 영역에 해당되는 contour 배열 정렬
- contour의 크기와 비율을 번호판 속 글자 크기/비율과 비슷한 조건으로 설정




[pic1]: https://user-images.githubusercontent.com/74947395/165599991-a911aa13-2bd4-44e9-b07e-9e250b60bf47.png
[pic2]: https://user-images.githubusercontent.com/74947395/165599987-faf5a444-29e3-47f0-9d70-eb7dfd01c573.png
[pic3]: https://user-images.githubusercontent.com/74947395/165599982-4b3fddbd-28e2-4022-bce5-5846339e3534.png
[pic4]: https://user-images.githubusercontent.com/74947395/165599972-3c013dbb-3265-4705-a348-ae40a075edce.png
[pic5]: https://user-images.githubusercontent.com/74947395/165600008-191105af-cd29-4170-afa0-297ed293b535.png
[pic6]: https://user-images.githubusercontent.com/74947395/165600007-495ed062-1b4e-427c-9e7d-3719858f3472.png
[pic7]: https://user-images.githubusercontent.com/74947395/165600004-1d1efcd6-b2d8-488c-9ebe-d1261d7f09dd.png
[pic8]: https://user-images.githubusercontent.com/74947395/165599994-37c2202b-50a8-4e76-ab15-2ad49154dca9.png

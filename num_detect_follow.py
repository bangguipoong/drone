import cv2
import numpy as np
import pytesseract
import time
from djitellopy import tello

me = tello.Tello()  # tello.Tello()을 me로 정의
me.connect()  # 텔로드론 연결
print(me.get_battery())
me.streamon()

room = [310,309,308,307,306]

room_num = int(input('안내받으실 호실을 입력하세요(313~325): '))

while not (room_num in room):
    room_num = int(input('해당하는 호실은 없습니다.\n안내받으실 호실을 입력하세요(313~325): '))

me.takeoff()


def Alt_Hold():
    while me.get_height() < 130:
        # print('height: ', me.get_height())
        me.send_rc_control(0, 0, 30, 0)
        time.sleep

def Yaw_Hold():
    yaw = me.get_yaw()
    # print('yaw:', yaw)

    if yaw > 0:
        me.send_rc_control(0,0,0,-(me.get_yaw()))

    elif yaw < 0:
        me.send_rc_control(0,0,0,-(me.get_yaw()))


while me.get_height() < 120:
    print('height: ', me.get_height())
    me.send_rc_control(0, 0, 20, 0)

if 313 <= room_num <= 321:  # 313호 앞
    me.rotate_counter_clockwise(91)
    me.move_forward(500)
    me.move_forward(50)

# else:  # 325호 앞
#     me.move_forward(140)
#     me.rotate_counter_clockwise(91)
#     me.move_forward(500)
#     me.move_forward(500)

global img

# 하이퍼 파라미터
Min_Area = 100  # 윤곽선의 최소 넓이
Max_Area = 2000  # 윤곽선의 최대 넓이
Min_Width, Min_Height = 1, 8  # 윤곽선의 최소 너비와 높이
Min_Ratio, Max_Ratio = 0.3, 1.0  # 가로세로 최대,최소 비율

Max_Diag_Multiplyer = 1  # contours의 간격
Max_Angle_Diff = 5.0  # contours의 각도
Max_Area_Diff = 1  # contours의 넓이 차이
Max_Width_Diff = 20  # contours의 너비 차이
Max_Height_Diff = 30  # contours의 높이 차이
Min_N_Matched = 3  # 위 조건의 3개 이상 맞으면 통과

pError_lr = 0
pid_lr = [0.1, 0.1, 0]

count = 0


def Img_Process():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_blurred = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)  # GaussianBlur는 img의 noisy을 줄여준다

    img_thresh = cv2.adaptiveThreshold(img_blurred, maxValue=255.0,
                                       adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       thresholdType=cv2.THRESH_BINARY_INV, blockSize=17, C=4)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(img_thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    # 사각형 선택
    possible_contours = []

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        if Max_Area > area > Min_Area and d['w'] > Min_Width and d['h'] > Min_Height and Min_Ratio < ratio < Max_Ratio:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    return possible_contours, img_thresh


# 해당하는 글자에 사각형 치기
def find_chars(contour_list):
    matched_result_idx = []

    for d1 in contour_list:
        matched_contours_idx = []

        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue
            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            # 거리 구하기
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))

            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w'] / d1['w'])
            height_diff = abs(d1['h'] - d2['h'] / d1['h'])

            if distance < diagonal_length1 * Max_Diag_Multiplyer \
                    and angle_diff < Max_Angle_Diff and area_diff < Max_Area_Diff \
                    and width_diff < Max_Width_Diff and height_diff < Max_Height_Diff:
                matched_contours_idx.append(d2['idx'])

        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < Min_N_Matched:
            continue

        matched_result_idx.append(matched_contours_idx)
        break

    return matched_result_idx


def Img_find_chars_rectangle(result_idx, possible_contours):
    matched_result = []

    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    return matched_result


def Img_Focusing_Cut(matched_result, img_thresh):
    global sorted_chars
    plate_imgs = []
    plate_infos = []

    for matched_chars in matched_result:
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x'])
    sum_height = 0

    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars))

    img_cropped = cv2.getRectSubPix(img_thresh, patchSize=(int(plate_width), int(plate_height)),
                                    center=(int(plate_cx), int(plate_cy))
                                    )

    plate_imgs.append(img_cropped)
    plate_infos.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })
    return plate_imgs, plate_infos


def Read_chars(plate_imgs):
    global result_num
    len_text = 3

    for i, plate_img in enumerate(plate_imgs):
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)

        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            area = w * h
            ratio = w / h

            if area > Min_Area and w > Min_Width and h > Min_Height \
                    and Min_Ratio < ratio < Max_Ratio:
                if x < plate_min_x:
                    plate_min_x = x
                if y < plate_min_y:
                    plate_min_y = y
                if x + w > plate_max_x:
                    plate_max_x = x + w
                if x > plate_max_y:
                    plate_max_y = y + h

        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

        # img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
        _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10,
                                        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

        chars = pytesseract.image_to_string(img_result, config='--psm 6')

        result_chars = ''

        for c in chars:
            if c.isdigit():
                result_chars += c
        #
        # if len_text == len(result_chars):
        result_num = result_chars

    return result_num


def Final_Rectangle(plate_infos, img):
    info = plate_infos[0]

    img_out = img.copy()

    cv2.rectangle(img_out, pt1=(info['x'], info['y']), pt2=(info['x'] + info['w'], info['y'] + info['h']),
                  color=(255, 0, 0), thickness=2)
    return img_out


def Move_go(number, count):
    me.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)

    if number == room_num:
        count += 1
    else:
        count = 0

    if count <= 1:
        me.send_rc_control(0, 50, 0, 0)

    else:
        if 306 <= room_num <= 310:
            me.send_rc_control(0, 0, 0, 0)
            me.rotate_clockwise(180)
            me.flip_left()
            me.takeoff()

        else:
            me.send_rc_control(0, 0, 0, 0)
            me.rotate_counter_clockwise(180)
            me.flip_left()
            me.takeoff()

    return count


def track_num_right(info, pid_lr, pError_lr):
    info = info[0]
    x = info['x']

    error_lr = x - 300
    print('error_lr:', error_lr)

    speed_lr = pid_lr[0] * error_lr + pid_lr[1] * (error_lr - pError_lr)
    speed_lr = int(np.clip(speed_lr, -50, 50))

    me.send_rc_control(speed_lr, 0, 0, 5)

    return error_lr


def track_num_left(info, pid_lr, pError_lr):
    info = info[0]
    x = info['x']

    error_lr = x - 200
    print('error_lr:', error_lr)

    speed_lr = pid_lr[0] * error_lr + pid_lr[1] * (error_lr - pError_lr)
    speed_lr = int(np.clip(speed_lr, -50, 50))

    me.send_rc_control(speed_lr, 0, 0, 5)

    return error_lr


while True:
    Alt_Hold()

    Yaw_Hold()

    img = me.get_frame_read().frame
    height, width, channel = img.shape

    possible_contours, img_thresh = Img_Process()

    result_idx = find_chars(possible_contours)

    matched_result = Img_find_chars_rectangle(result_idx, possible_contours)

    if len(matched_result) > 0:
        plate_imgs, plate_infos = Img_Focusing_Cut(matched_result, img_thresh)

        num = Read_chars(plate_imgs)

        print(num)
        if 301 <= room_num <= 305:
            track_num_left(plate_infos, pid_lr, pError_lr)
        else:
            track_num_right(plate_infos, pid_lr, pError_lr)

        count = Move_go(num, count)

        cv2.imshow('tello_cam', Final_Rectangle(plate_infos, img))

    else:
        fb_speed = 30
        me.send_rc_control(0, fb_speed, 0, 0)
        img_out = img.copy()
        cv2.imshow('tello_cam', img_out)

    # 27=esc ,b = battery
    K = cv2.waitKey(1) & 0XFF
    if K == 27:
        me.land()
        break

    elif K == ord('b'):
        print(me.get_battery())
import math
import random
import cv2
import numpy as np

WIDTH = 800
HEIGHT = 800

XCEN = WIDTH // 2
YCEN = HEIGHT // 2
RADIUS = 300

BARLENGTH = 120
BARTHICC = 20
MIN_RED_AREA = 500
SMOOTHING_FACTOR = 0.2
INITIAL_BALL_X_SPEED = 4
INITIAL_BALL_Y_SPEED = -3
BASE_BALL_SPEED = math.hypot(INITIAL_BALL_X_SPEED, INITIAL_BALL_Y_SPEED)
CORRECT_HIT_SPEEDUP = 1.1
INCORRECT_HIT_SLOWDOWN = 0.5

BACKGROUND = (18, 16, 28)
WHITE = (235, 235, 240)
MUTED = (155, 155, 175)
ARENA_COLOR = (90, 85, 105)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
PANEL = (32, 29, 45)

selfangle = 90
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam. Check that it is connected and available.")

ballx = XCEN
bally = YCEN
BALLRAD = 10
BALLTARGET = random.choice(["self", "mirror"])
ASSURANCE = 3
BALLTOUCHED = False
CERTAINTY = 0

ballxspd = INITIAL_BALL_X_SPEED
ballyspd = INITIAL_BALL_Y_SPEED


def draw_text(screen, text, y, scale, color, thickness=1):
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (WIDTH - size[0]) // 2
    cv2.putText(
        screen,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

def get_bar_pnts(angle):
    radians = math.radians(angle)
    barxcen = int(XCEN + RADIUS * math.cos(radians))
    barycen = int(YCEN + RADIUS * math.sin(radians))

    tangent = math.radians(angle + 90)

    dx = (BARLENGTH // 2) * math.cos(tangent)
    dy = (BARLENGTH // 2) * math.sin(tangent)

    pnt1 = (int(barxcen - dx), int(barycen - dy))
    pnt2 = (int(barxcen + dx), int(barycen + dy))

    return pnt1, pnt2

def dist_to_bar(ballx, bally, pnt1, pnt2):
    a = np.array(pnt1, dtype=float)
    b = np.array(pnt2, dtype=float)
    p = np.array([ballx, bally], dtype=float)

    ab = b - a

    lensq = np.dot(ab, ab)
    t = np.dot(p - a, ab) / lensq

    t = max(0, min(1, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def random_bounce():
    global ballxspd, ballyspd
    speed = math.sqrt(ballxspd**2 + ballyspd**2)
    dx = ballx - XCEN
    dy = bally - YCEN

    inward_angle = math.atan2(-dy, -dx)
    angle = inward_angle + math.radians(random.uniform(-30, 30))

    ballxspd = speed * math.cos(angle)
    ballyspd = speed * math.sin(angle)

def adjust_ball_speed(multiplier, minimum=BASE_BALL_SPEED):
    global ballxspd, ballyspd
    speed = math.hypot(ballxspd, ballyspd)
    new_speed = max(minimum, speed * multiplier)
    ballxspd *= new_speed / speed
    ballyspd *= new_speed / speed


while True:
    screen = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    cv2.circle(screen, (XCEN, YCEN - 70), 190, PANEL, 2, cv2.LINE_AA)
    cv2.circle(screen, (XCEN, YCEN - 70), 165, PANEL, 1, cv2.LINE_AA)
    draw_text(screen, "PONG", 300, 1.8, WHITE, 3)
    draw_text(screen, "You are filled with anger.", 345, 0.65, RED, 2)
    draw_text(screen, "Haunted by guilt.", 380, 0.65, BLUE, 2)
    draw_text(screen, "SPACE  START", 545, 0.65, RED, 2)
    draw_text(screen, "Q  QUIT", 585, 0.55, MUTED)
    cv2.imshow("Game", screen)
    key = cv2.waitKey(20) & 0xFF
    if key == ord(' '):
        break
    if key == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        raise SystemExit

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 80, 60])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 80, 60])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= MIN_RED_AREA:
                x, y, w, h = cv2.boundingRect(largest)
                hand_x = x + w // 2
                hand_y = y + h // 2
                frame_height, frame_width = frame.shape[:2]
                cam_xcen = frame_width // 2
                cam_ycen = frame_height // 2
                dx = hand_x - cam_xcen
                dy = hand_y - cam_ycen
                hand_angle = math.degrees(math.atan2(dy, dx)) % 360
                angle_diff = (hand_angle - selfangle + 180) % 360 - 180
                selfangle = (selfangle + SMOOTHING_FACTOR * angle_diff) % 360

    screen = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    ballx += ballxspd
    bally += ballyspd
    dx = ballx - XCEN
    dy = bally - YCEN
    distance = math.sqrt(dx**2 + dy**2)
    if distance + BALLRAD >= RADIUS:
        nx = dx / distance
        ny = dy / distance
        dot = ballxspd * nx + ballyspd * ny
        ballxspd -= 2 * dot * nx
        ballyspd -= 2 * dot * ny
        ballx = XCEN + (RADIUS - BALLRAD) * nx
        bally = YCEN + (RADIUS - BALLRAD) * ny

    cv2.circle(screen, (XCEN, YCEN), RADIUS, ARENA_COLOR, 2, cv2.LINE_AA)

    p1, p2 = get_bar_pnts(selfangle)
    cv2.line(screen, p1, p2, RED, BARTHICC, cv2.LINE_AA)

    mirangle = (selfangle + 180) % 360
    p1, p2 = get_bar_pnts(mirangle)
    cv2.line(screen, p1, p2, BLUE, BARTHICC, cv2.LINE_AA)
    if BALLTARGET == "self":
        ballcolor = RED
    else:
        ballcolor = BLUE

    cv2.circle(screen, (int(ballx), int(bally)), BALLRAD + 2, WHITE, -1, cv2.LINE_AA)
    cv2.circle(screen, (int(ballx), int(bally)), BALLRAD, ballcolor, -1, cv2.LINE_AA)

    selfp1, selfp2 = get_bar_pnts(selfangle)
    mirangle = (selfangle + 180) % 360
    mirp1, mirp2 = get_bar_pnts(mirangle)
    self_dist = dist_to_bar(ballx, bally, selfp1, selfp2)
    mir_dist = dist_to_bar(ballx, bally, mirp1, mirp2)

    selftouched = self_dist <= BALLRAD + BARTHICC / 2
    mirtouched = mir_dist <= BALLRAD + BARTHICC / 2

    touchingbar = selftouched or mirtouched
    if touchingbar and not BALLTOUCHED:
        if selftouched:
            if BALLTARGET == "self":
                CERTAINTY += 1
                random_bounce()
                adjust_ball_speed(CORRECT_HIT_SPEEDUP)
                BALLTARGET = random.choice(["self", "mirror"])
            else:
                ASSURANCE -= 1
                adjust_ball_speed(INCORRECT_HIT_SLOWDOWN)
                BALLTARGET = random.choice(["self", "mirror"])
        elif mirtouched:
            if BALLTARGET == "mirror":
                CERTAINTY += 1
                random_bounce()
                adjust_ball_speed(CORRECT_HIT_SPEEDUP)
                BALLTARGET = random.choice(["self", "mirror"])
            else:
                ASSURANCE -= 1
                adjust_ball_speed(INCORRECT_HIT_SLOWDOWN)
                BALLTARGET = random.choice(["self", "mirror"])

    BALLTOUCHED = touchingbar
    if ASSURANCE <= 0:
        cv2.rectangle(screen, (120, 275), (680, 525), PANEL, -1)
        cv2.rectangle(screen, (120, 275), (680, 525), ARENA_COLOR, 1)
        draw_text(screen, "OUT OF ASSURANCE", 340, 0.8, RED, 2)
        draw_text(screen, "Giving up on fighting yourself?", 385, 0.55, WHITE)
        draw_text(screen, f"FINAL CERTAINTY  {CERTAINTY}", 420, 0.55, MUTED)
        draw_text(screen, "SPACE  TRY AGAIN", 455, 0.6, WHITE, 2)
        draw_text(screen, "Q  QUIT", 490, 0.5, MUTED)
        cv2.imshow("Game", screen)
        key = cv2.waitKey(0) & 0xFF
        if key == ord(' '):
            ASSURANCE = 3
            CERTAINTY = 0
            BALLTARGET = random.choice(["self", "mirror"])
            ballx = XCEN
            bally = YCEN
            BALLTOUCHED = False
            ballxspd = INITIAL_BALL_X_SPEED
            ballyspd = INITIAL_BALL_Y_SPEED
            selfangle = 90
            continue
        elif key == ord('q'):
            break
        else:
            continue

    cv2.rectangle(screen, (0, 0), (WIDTH, 76), PANEL, -1)
    cv2.line(screen, (0, 76), (WIDTH, 76), ARENA_COLOR, 1, cv2.LINE_AA)
    cv2.putText(screen, f"ASSURANCE  {ASSURANCE}", (24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2, cv2.LINE_AA)
    cv2.putText(screen, f"CERTAINTY  {CERTAINTY}", (24, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MUTED, 1, cv2.LINE_AA)
    speed_ratio = math.hypot(ballxspd, ballyspd) / BASE_BALL_SPEED
    cv2.putText(screen, f"PACE  {speed_ratio:.1f}x", (340, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(screen, "MATCH THE BALL", (WIDTH - 210, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MUTED, 1, cv2.LINE_AA)

    cv2.imshow("Game", screen)
    key = cv2.waitKey(20) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import json
import os
import sys

# ── CONFIG ──────────────────────────────────────────────────────────
LEFT_DIR   = "left"
RIGHT_DIR  = "right"
SAVE_FILE  = "training_data.json"
DISP_W     = 800    # display width per panel (px); clicks are scaled back to full res
DISP_H     = 600    # display height per panel (px)

# ── STATE ───────────────────────────────────────────────────────────
class State:
    left_img = None
    right_img = None
    combined = None
    left_w = 0
    clicks = []          # [(x,y)] in original image coords
    current_pair = {}    # {"left": (x,y), "right": (x,y)}
    all_data = []        # list of {left_file, right_file, left_pt, right_pt, distance}
    waiting_for = "left" # which image we expect a click on next
    pair_index = 0
    pairs = []           # list of (left_path, right_path)
    scale = 1.0

WINDOW = "Click Points  |  LEFT ←  → RIGHT  |  Q=quit  N=next pair  D=done with this image"

def load_existing_data():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(f"[warn] {SAVE_FILE} is empty or corrupt — starting fresh")
    return []

def save_data(data):
    with open(SAVE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[saved] {len(data)} point-pairs → {SAVE_FILE}")

def build_pairs():
    """Match left/right images by sorted order or common name fragments."""
    left_files  = sorted(os.listdir(LEFT_DIR))
    right_files = sorted(os.listdir(RIGHT_DIR))

    # Try matching by replacing left/right in filenames
    pairs = []
    used_right = set()
    for lf in left_files:
        # Try direct name matching: left1 <-> right1, newleft <-> newright
        rname = lf.replace("left", "right").replace("LEFT", "RIGHT")
        if rname in right_files and rname not in used_right:
            pairs.append((os.path.join(LEFT_DIR, lf), os.path.join(RIGHT_DIR, rname)))
            used_right.add(rname)
        else:
            # fallback: pair by index
            pass

    # Any unmatched right files, pair remaining by index
    if not pairs:
        for lf, rf in zip(left_files, right_files):
            pairs.append((os.path.join(LEFT_DIR, lf), os.path.join(RIGHT_DIR, rf)))

    return pairs

def show_combined(state):
    """Display left and right images side by side with clicked points drawn."""
    combined = state.combined.copy()
    h, w = combined.shape[:2]

    # Draw divider line
    cv2.line(combined, (state.left_w, 0), (state.left_w, h), (255, 255, 0), 2)

    # Draw existing clicks for current image pair
    s = state.scale if state.scale > 0 else 1.0
    for entry in state.all_data:
        if entry["left_file"] == state.pairs[state.pair_index][0]:
            lx, ly = entry["left_pt"][0] * s, entry["left_pt"][1] * s
            rx, ry = entry["right_pt"][0] * s, entry["right_pt"][1] * s
            cv2.circle(combined, (int(lx), int(ly)), 6, (0, 255, 0), -1)
            cv2.putText(combined, f'{entry["distance"]:.2f}m', (int(lx)+8, int(ly)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(combined, (int(rx) + state.left_w, int(ry)), 6, (0, 255, 0), -1)

    # Draw pending click
    if "left" in state.current_pair:
        lx, ly = state.current_pair["left"][0] * s, state.current_pair["left"][1] * s
        cv2.circle(combined, (int(lx), int(ly)), 6, (0, 0, 255), -1)
        cv2.putText(combined, "LEFT ✓ → click RIGHT", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Status bar
    status = f"Pair {state.pair_index+1}/{len(state.pairs)}  |  "
    if state.waiting_for == "left":
        status += "Click LEFT image point"
    else:
        status += "Click RIGHT image point"
    status += f"  |  {len(state.all_data)} points collected"

    cv2.putText(combined, status, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow(WINDOW, combined)

def on_mouse(event, x, y, flags, param):
    state = param
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    s = state.scale if state.scale > 0 else 1.0
    if state.waiting_for == "left":
        if x < state.left_w:
            fx, fy = x / s, y / s          # full-resolution coords
            state.current_pair["left"] = (fx, fy)
            state.waiting_for = "right"
            print(f"  Left point:  ({fx:.0f}, {fy:.0f})  →  now click the SAME object on the RIGHT image")
        else:
            print("  ⚠ Click on the LEFT image first!")
    elif state.waiting_for == "right":
        if x >= state.left_w:
            rx = (x - state.left_w) / s    # full-resolution right-image coords
            ry = y / s
            state.current_pair["right"] = (rx, ry)
            print(f"  Right point: ({rx}, {ry})")
            # Ask for distance
            ask_distance(state)
        else:
            print("  ⚠ Now click on the RIGHT image!")

    show_combined(state)

def ask_distance(state):
    """Prompt user in terminal for the known distance."""
    while True:
        try:
            d = input("  Enter known distance to this object (in meters): ").strip()
            if d == "":
                print("  Cancelled this point pair.")
                state.current_pair = {}
                state.waiting_for = "left"
                return
            dist = float(d)
            break
        except ValueError:
            print("  Please enter a number (or press Enter to cancel).")

    entry = {
        "left_file":  state.pairs[state.pair_index][0],
        "right_file": state.pairs[state.pair_index][1],
        "left_pt":    list(state.current_pair["left"]),
        "right_pt":   list(state.current_pair["right"]),
        "distance":   dist
    }
    state.all_data.append(entry)
    print(f"  ✓ Saved! ({len(state.all_data)} total point-pairs)")
    save_data(state.all_data)

    # Reset for next point
    state.current_pair = {}
    state.waiting_for = "left"

def load_pair(state):
    lpath, rpath = state.pairs[state.pair_index]
    left  = cv2.imread(lpath)
    right = cv2.imread(rpath)
    if left is None:
        print(f"ERROR: cannot read {lpath}")
        sys.exit(1)
    if right is None:
        print(f"ERROR: cannot read {rpath}")
        sys.exit(1)

    # Scale factor: full-res → display
    lh, lw = left.shape[:2]
    state.scale  = min(DISP_W / lw, DISP_H / lh)
    dw = int(lw * state.scale)
    dh = int(lh * state.scale)

    left_d  = cv2.resize(left,  (dw, dh))
    right_d = cv2.resize(right, (dw, dh))

    state.left_img  = left_d
    state.right_img = right_d
    state.left_w    = dw
    state.combined  = np.hstack([left_d, right_d])
    state.current_pair = {}
    state.waiting_for = "left"

    print(f"\n{'='*60}")
    print(f"Image pair {state.pair_index+1}/{len(state.pairs)}")
    print(f"  Left:  {lpath}")
    print(f"  Right: {rpath}")
    print(f"Click corresponding points on LEFT then RIGHT, enter distance.")
    print(f"Keys: N=next pair  P=prev pair  U=undo last  Q=quit")
    print(f"{'='*60}")

def main():
    state = State()
    state.pairs = build_pairs()
    state.all_data = load_existing_data()

    if not state.pairs:
        print("No image pairs found! Put images in left/ and right/ directories.")
        sys.exit(1)

    print(f"Found {len(state.pairs)} image pair(s):")
    for i, (l, r) in enumerate(state.pairs):
        print(f"  {i+1}. {l}  ↔  {r}")

    load_pair(state)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse, state)
    show_combined(state)

    while True:
        key = cv2.waitKey(50) & 0xFF

        if key == ord('q'):
            print(f"\nDone! {len(state.all_data)} point-pairs saved to {SAVE_FILE}")
            break

        elif key == ord('n'):
            # Next pair
            if state.pair_index < len(state.pairs) - 1:
                state.pair_index += 1
                load_pair(state)
                show_combined(state)
            else:
                print("  Already at last pair.")

        elif key == ord('p'):
            # Previous pair
            if state.pair_index > 0:
                state.pair_index -= 1
                load_pair(state)
                show_combined(state)
            else:
                print("  Already at first pair.")

        elif key == ord('u'):
            # Undo last point
            if state.current_pair:
                state.current_pair = {}
                state.waiting_for = "left"
                print("  Undid pending click.")
                show_combined(state)
            elif state.all_data:
                removed = state.all_data.pop()
                save_data(state.all_data)
                print(f"  Undid last saved point: {removed['left_pt']} ↔ {removed['right_pt']}")
                show_combined(state)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

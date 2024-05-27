import cv2
from CocoInfo import CocoInfo


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)


def plot_bboxes(image, boxes, labels=None, colors=None, score=True, conf=None, filter_labels: set[str] | None = None):
    coco_info = CocoInfo()
    if not labels:
        labels = coco_info.get_coco_labels()

    if not colors:
        colors = coco_info.get_coco_colors_pattern()

    # Process each box
    for box in boxes:
        label_index = int(box[-1])
        box_label_name = labels[label_index + 1]
        box_score = float(box[-2])

        if conf is not None and box_score <= conf:
            continue

        # Check against filter_labels if it is provided
        if filter_labels is not None and box_label_name not in filter_labels:
            continue

        display_label = f"{box_label_name} {round(100 * box_score, 1)}%" if score else box_label_name
        color = colors[label_index]

        # Draw the box and label on the image
        box_label(image, box[:4], display_label, color)

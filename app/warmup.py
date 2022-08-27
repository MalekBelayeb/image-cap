from generate_caption import generate_caption_greedy
import logging 

b64_img_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAAl21bKAAAAA1BMVEX_TQBcNTh_AAAAAXRSTlPM0jRW_QAAAApJREFUeJxjYgAAAAYAAzY3fKgAAAAASUVORK5CYII="
caption = generate_caption_greedy(b64_img_str)
logging.error(f'warm up 1x1 png test: {caption}')


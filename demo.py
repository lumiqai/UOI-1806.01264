from model import build_model


TAG_O, TAG_B, TAG_I, TAG_E = 0, 1, 2, 3


model = build_model(token_num=100,
                    tag_num=4)
model.summary(line_length=80)

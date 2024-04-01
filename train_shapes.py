from fastai.vision.all import *
from os.path import *
import os

avg_stats_255 = [[128, 128, 128], [128, 128, 128]]
fig, axs = plt.subplots(nrows=6, ncols=10)
shapes_path = Path(r"C:\Users\piotr\source\PROJEKTY\PY\MpPlots\out")
label_to_float = {"circle": 0.0, "square": 1.0, "triangle": 2.0}


class ShowAfterBatch(Callback):
    def __init__(self):
        self.count = 0

    def after_batch(self):
        if self.count < 60:
            t1 = TensorImage(self.x[0].clone())
            t1.show(ctx=axs[self.count // 10, self.count % 10])
        self.count += 1


def show_predictions(learner):
    print("--- Predictions ---")
    test_folder = Path(r"C:\Users\piotr\source\PROJEKTY\PY\MpPlots\testem\triangle")
    for img_name in os.listdir(test_folder):
        test_img = PILImage.create(test_folder / img_name)
        print(learner.predict(test_img))

def create_image_to_float_dls(
    shapes_path,
    train,
    valid_pct,
    item_tfms,
    batch_tfms,
    bs=5,
    verbose=True,
    num_workers=0,
    seed=0,
    **kwargs
):
    splitter = RandomSplitter(valid_pct, seed=seed)
    get_items = get_image_files
    dblock = DataBlock(
        blocks=(ImageBlock(PILImage), RegressionBlock(n_out=1)),
        get_items=get_items,
        splitter=splitter,
        get_y=lambda x: label_to_float[parent_label(x)],
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )
    return DataLoaders.from_dblock(
        dblock,
        shapes_path,
        path=shapes_path,
        bs=bs,
        verbose=verbose,
        num_workers=num_workers,
        **kwargs
    )


def train_categories(show_after_batch=False, show_predictions=True):
    if __name__ == "__main__":
        multiprocessing.freeze_support()
        shapes_dls = ImageDataLoaders.from_folder(
            shapes_path,
            train="train",
            valid_pct=0.2,
            item_tfms=Resize(224),
            batch_tfms=[
                *aug_transforms(size=128),
                Normalize.from_stats(*imagenet_stats),
            ],
            bs=5,
            verbose=True,
            num_workers=0,
            y_range=(0.0, 4.0),
        )

        cbs = [TrainEvalCallback, Recorder]
        if show_after_batch:
            testcbk = ShowAfterBatch()
            cbs.append(testcbk)

        shapes_learner = vision_learner(
            shapes_dls, resnet34, metrics=error_rate, cbs=cbs
        )

        shapes_learner.fit_one_cycle(1, 0.001)

        if show_after_batch:
            plt.show()
        shapes_learner.show_results(max_n=20)

        if show_predictions:
            print("--- Predictions ---")
            test_folder = Path(
                r"C:\Users\piotr\source\PROJEKTY\PY\MpPlots\testem\triangle"
            )
            for img_name in os.listdir(test_folder):
                test_img = PILImage.create(test_folder / img_name)
                print(shapes_learner.predict(test_img))

            # shapes_learner.save("shapes_learner.model", with_opt=True)


# train_categories()
regression_dls = create_image_to_float_dls(
    shapes_path,
    train="train",
    valid_pct=0.2,
    item_tfms=Resize(224),
    batch_tfms=[
        *aug_transforms(size=128),
        Normalize.from_stats(*imagenet_stats),
    ],
)

shapes_learner = vision_learner(regression_dls, resnet34, metrics=mse)
shapes_learner.fit_one_cycle(25, 0.01)
show_predictions(shapes_learner)

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.losses import binary_crossentropy

smooth = 1.0

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def unet(pretrained_weights=0, input_size=(512, 512, 1)):
    inputs = Input(input_size)
    #512

    down1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    down1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(down1)
    down1_pool = MaxPooling2D(pool_size=(2, 2))(down1)

    down2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(down1_pool)
    down2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(down2)
    down2_pool = MaxPooling2D(pool_size=(2, 2))(down2)
    #128

    down3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(down2_pool)
    down3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(down3)
    down3_pool = MaxPooling2D(pool_size=(2, 2))(down3)

    down4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(down3_pool)
    down4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(down4)
    drop4 = Dropout(0.5)(down4)
    down4_pool = MaxPooling2D(pool_size=(2, 2))(drop4)

    center = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(down4_pool)
    center = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(center)
    drop5 = Dropout(0.5)(center)

    up4 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    up4_merge = merge([drop4, up4], mode='concat', concat_axis=3)
    up4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up4_merge)
    up4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up4)

    up3 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(up4))
    up3_merge = merge([down3, up3], mode='concat', concat_axis=3)
    up3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up3_merge)
    up3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up3)

    up2 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(up3))
    up2_merge = merge([down2, up2], mode='concat', concat_axis=3)
    up2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up2_merge)
    up2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up2)

    up1 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(up2))
    up1_merge = merge([down1, up1], mode='concat', concat_axis=3)
    up1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up1_merge)
    up1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up1)
    up1 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up1)
    Classify = Conv2D(1, 1, activation='sigmoid')(up1)

    model = Model(input=inputs, output=Classify)

    model.compile(optimizer=Adam(lr=1e-4), loss=bce_dice_loss, metrics=[dice_coef])

    if (pretrained_weights):
        print("yes")
        model.load_weights(pretrained_weights)

    return model

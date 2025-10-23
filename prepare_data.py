"""Utilities for preparing bot detection datasets."""

# ruff: noqa: RUF003

from pathlib import Path

import pandas as pd

DATA_PATH = Path("data")


def prepare_data(data_path: str = DATA_PATH) -> None:
    ###############################################################################
    # Шаг 1. Получаем данные
    ###############################################################################
    # json такой структуры можно прочитать, если при вызове функции
    # pd.read_json в параметр orient передать значение "index",
    # либо в параметр lines - True (но не оба сразу).
    # В 1 случае чтение каждое значения уровня root станет отедльным индексом, а
    # все значения на уровень ниже станут столбцами. С такой структурой не будет
    # не очень удобно работать, т.к. получится разряженный датафрейм ввиду того,
    # что у диалогов в исходном json'е сильно разнится количество сообщений: от 4
    # до 252 - если интересно, попробуйте прочитать так и посмотреть на структуру.
    # Во 2 случае чтение json'а будет происходить построчно, причём отдельная
    # строка будет записываться в отедльный столбец - получим pandas.DataFrame из
    # одной строки. С такой таблицей проще работать - так и прочтём.
    train_raw = pd.read_json(data_path / "train.json", lines=True)  # orient="index")
    test_raw = pd.read_json(data_path / "test.json", lines=True)  # orient="index")

    df_classes_train = pd.read_csv(data_path / "ytrain.csv")
    pd.read_csv(data_path / "ytest.csv")

    ###############################################################################
    # Шаг 2. Подготавливаем данные
    ###############################################################################
    # # 2.1 Преобразуем вид датафреймов
    # ------------------------------------------------------------>>>
    train_dialogs = (
        train_raw.iloc[0]  # превращаем pandas.DataFrame из одной строки в pandas.Series
        .rename("dialog")  # имя полученной серии - "0" (по индексу строки pandas.DataFrame, из которой она
        # получилась) - переименовываем
        .reset_index()  # ресетим индекс, чтобы id диалогов из индекса стали столбцом; при этом снова получаем
        # pandas.DataFrame, т.к. у объекта будет уже два столбца, а у серии может быть только один
        .rename({"index": "dialog_id"}, axis=1)  # либо .rename(columns={'index': 'dialog_id'})
    )

    labels_wide = (
        df_classes_train.pivot_table(
            index="dialog_id",
            columns="participant_index",
            values="is_bot",
            aggfunc="first",
        )
        .rename(columns={0: "participant_0_is_bot", 1: "participant_1_is_bot"})
        .reset_index()
    )

    # Мёрджим трейн-датасет с лэйблами
    train_df = train_dialogs.merge(labels_wide, on="dialog_id", how="left")

    test_df = test_raw.iloc[0].rename("dialog").reset_index().rename(columns={"index": "dialog_id"})
    # <<<------------------------------------------------------------

    # # 2.2 Разделим теперь сообщения по участникам диалогов
    # ------------------------------------------------------------>>>
    # Сначала определим сообщения участника с индексом "0":
    train_df["participant_0_messages"] = train_df.apply(
        lambda row: [  # для каждой строки (т.е. для каждого диалога) хотим получить список сообщений
            msg["text"]  # берём текст сообщения
            for msg in row.dialog  # пробегаем по всем сообщеням диалогоа
            if msg["participant_index"] == "0"  # берём, если индекс участника равен "0"
        ],
        axis=1,  # применяем функцию к столбцам, т.е. пробегаем по строкам
        # (по дефолту стотит axis=0, что соответсвует пробеганию по столбцам)
    )
    train_df["participant_1_messages"] = train_df.apply(
        lambda row: [msg["text"] for msg in row.dialog if msg["participant_index"] == "1"], axis=1
    )
    test_df["participant_0_messages"] = test_df.apply(
        lambda row: [msg["text"] for msg in row.dialog if msg["participant_index"] == "0"], axis=1
    )
    test_df["participant_1_messages"] = test_df.apply(
        lambda row: [msg["text"] for msg in row.dialog if msg["participant_index"] == "1"], axis=1
    )

    # Теперь каждый список сообщенй объединим в одно сообщение (можно было сделать во время прошлой операции)
    # Операция пременяется к конкретному столбцу, поэтому можно не указывать axis=1
    train_df["participant_0_messages"] = train_df["participant_0_messages"].apply(lambda row: " ".join(row)).fillna("")
    train_df["participant_1_messages"] = train_df["participant_1_messages"].apply(lambda row: " ".join(row)).fillna("")
    test_df["participant_0_messages"] = test_df["participant_0_messages"].apply(lambda row: " ".join(row)).fillna("")
    test_df["participant_1_messages"] = test_df["participant_1_messages"].apply(lambda row: " ".join(row)).fillna("")
    # <<<------------------------------------------------------------

    # # 2.3 Получим итоговые данные
    # ------------------------------------------------------------>>>
    # Трейн-данные приводим к виду:
    # ID — комбинированное поле (dialog_id_participantIndex)
    # text — строка, объединяющая в себе все сообщения участника
    # is_bot — метка класса
    train_df_0 = train_df.assign(ID=train_df.dialog_id + "_0").rename(
        {"participant_0_messages": "text", "participant_0_is_bot": "is_bot"}, axis=1
    )[["ID", "text", "is_bot"]]
    train_df_1 = train_df.assign(ID=train_df.dialog_id + "_1").rename(
        {"participant_1_messages": "text", "participant_1_is_bot": "is_bot"}, axis=1
    )[["ID", "text", "is_bot"]]
    train_prepared = pd.concat([train_df_0, train_df_1], ignore_index=True)

    # Тест-данные приводим к виду:
    # ID — комбинированное поле (dialog_id_participantIndex)
    # text — строка, объединяющая в себе все сообщения участника
    test_df_0 = test_df.assign(ID=test_df.dialog_id + "_0").rename({"participant_0_messages": "text"}, axis=1)[
        ["ID", "text"]
    ]
    test_df_1 = test_df.assign(ID=test_df.dialog_id + "_1").rename({"participant_1_messages": "text"}, axis=1)[
        ["ID", "text"]
    ]
    test_prepared = pd.concat([test_df_0, test_df_1], ignore_index=True)

    ###############################################################################
    # Шаг 3. Сохраняем подготовленные данные
    ###############################################################################
    train_prepared.to_csv(data_path / "train_prepared.csv")
    test_prepared.to_csv(data_path / "test_prepared.csv")


if __name__ == "__main__":
    prepare_data()

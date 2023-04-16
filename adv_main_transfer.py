# coding: utf-8
import time, sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Log import Logger  # 日志
from src.AdvUtil import ExeDataset
import copy

Logger.log("./logdata/")

# seed
np.random.seed(1)
torch.manual_seed(1)


def num2tensor(input_data, use_gpu, bool):  # list2tensor
    if bool == "long":
        output_data = torch.from_numpy(np.array([input_data]))
        output_data = (
            Variable(output_data.long(), requires_grad=False).cuda()
            if use_gpu
            else Variable(output_data.long(), requires_grad=False)
        )
    if bool == "float":
        input_data = torch.tensor(input_data)
        output_data = (
            Variable(input_data.float(), requires_grad=False).cuda()
            if use_gpu
            else Variable(input_data.float(), requires_grad=False)
        )
    return output_data


def select_tensor(input_tensor, dim, begin, end):  #
    index_list = [i for i in range(begin, end)]
    output_tensor = torch.index_select(
        input_tensor, dim=dim, index=torch.tensor(index_list)
    )
    return output_tensor


def embed2tabel(embed_item, use_gpu):  # get embed2tabel
    embed_item_list = []
    for i in range(1, 257):
        embed_item_list.append(embed_item)
    embed_item_tabel = num2tensor(embed_item_list, use_gpu, "float")
    # print(embed_item_tabel)
    return embed_item_tabel


def embed2data(
    data, index_list, new_embed_input, fixed_enmbed_tabel, use_gpu
):  # embeding to byte
    loss_func = nn.MSELoss(reduction="none")
    for j in range(0, len(index_list)):
        embed_item = (
            new_embed_input[0][:][index_list[j]].view(-1).detach().numpy().tolist()
        )
        embed_item_tabel = embed2tabel(embed_item, use_gpu)
        loss = torch.mean(loss_func(embed_item_tabel, fixed_enmbed_tabel), 1)
        loss_list = loss.view(-1).detach().numpy().tolist()
        minloss = min(loss_list)
        idx = loss_list.index(minloss)
        data[index_list[j]] = idx + 1
    return data


def datd2why(data, length):
    why_list = np.random.randint(0, 1, 257)
    for i in range(0, length):
        why_list[data[i]] += 1
    index_list = why_list[:].tolist()
    why_list.sort()
    one_list = []
    for i in range(1, len(index_list)):
        index_adv = [x for (x, y) in enumerate(index_list) if y == why_list[i]]
        # index_adv = index_list.index(why_list[i])
        for index_this in index_adv:
            if (index_this not in one_list) and index_this != 0 and len(one_list) < 10:
                one_list.append(index_this)
                continue
    one_list.sort()
    may_list = []
    interval = 3
    for index_this in one_list:
        if (index_this - interval > 0) and (index_this + interval < 257):
            may_list.append([index_this - interval, index_this + interval + 1])
    # may_list.sort()
    return may_list


def generate_label(use_gpu, label_length):  # select  gaol label
    label_list = []
    list0 = np.random.randint(0, 1, label_length)
    label0 = (
        Variable(torch.from_numpy(np.array(list0)).float(), requires_grad=False).cuda()
        if use_gpu
        else Variable(torch.from_numpy(np.array(list0)).float(), requires_grad=False)
    )
    label_list.append(label0)
    list1 = np.random.randint(1, 2, label_length)
    label1 = (
        Variable(torch.from_numpy(np.array(list1)).float(), requires_grad=False).cuda()
        if use_gpu
        else Variable(torch.from_numpy(np.array(list1)).float(), requires_grad=False)
    )
    # label_list.append(label1)
    return label_list


def data2adv(
    malconv, add_data, data, insert_index, adv_length, use_gpu
):  # initialization byte
    may_list = datd2why(data, length)
    data0 = data[0:insert_index]
    data2 = data[insert_index : first_n_byte - adv_length]
    add_data_list = []
    add_data_list.append(add_data)
    discount = len(may_list)
    for may_index in may_list:
        add_data = np.random.randint(may_index[0], may_index[1], adv_length)
        add_data_list.append(add_data)

    label_list = generate_label(use_gpu, discount + 1)
    max_list = []
    max_index_list = []
    for label in label_list:
        label = label.unsqueeze(1)
        adv_data_list = []
        temp_count_list = []
        for add_data in add_data_list:
            adv_data = np.concatenate([np.concatenate([data0, add_data]), data2])
            adv_data_list.append(adv_data)

        exe_input = num2tensor(adv_data_list, use_gpu, "long").squeeze(0)
        pred = malconv(exe_input)
        loss = bce_loss(pred, label)
        loss.backward()
        embed_sign = torch.sign(malconv.embed_x.grad.data)
        for i in range(0, discount + 1):
            temp_count = []
            for j in range(insert_index, insert_index + adv_length):
                adv_sign = embed_sign[i][j][:]
                zero_sign = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                if torch.eq(adv_sign, zero_sign).sum().item() != 8:
                    temp_count.append(j)
            temp_count_list.append(len(temp_count))
        max_count = max(temp_count_list)
        # print(max_count)
        max_index_list.append(temp_count_list.index(max_count))
        max_list.append(max_count)
    idx = max_list.index(max(max_list))
    label = Variable(torch.from_numpy(np.array([[idx]])).float(), requires_grad=False)
    adv_data = np.concatenate(
        [np.concatenate([data0, add_data_list[max_index_list[idx]]]), data2,]
    )
    return adv_data, label


if __name__ == "__main__":

    valid_label_path = "./data/train_test.csv"
    chkpt_acc_path = "./checkpoint/malconv.model"
    # valid_label_path = "./data/test_data2.csv"
    # chkpt_acc_path = "./checkpoint/large_malconv.model"

    batch_size = 1
    first_n_byte = 1000000
    # adv_length = 10000
    adv_step = 100
    npy_name = (
        str(sys.argv[0]).split(".")[0]
        + "_"
        + chkpt_acc_path.split(".")[1].split("/")[-1]
    )
    print(npy_name)
    adv_length = int(sys.argv[1])
    write_flag = int(sys.argv[2])
    # insert_index = 1024

    validloader = DataLoader(
        ExeDataset(valid_label_path, first_n_byte),
        batch_size=batch_size,
        shuffle=False,
        num_workers=64,
    )

    use_gpu = torch.cuda.is_available()
    use_gpu = False
    malconv = torch.load(chkpt_acc_path, map_location=torch.device("cpu"))
    sigmoid = nn.Sigmoid()
    bce_loss = nn.BCEWithLogitsLoss()

    if use_gpu:
        malconv = malconv.cuda()
        bce_loss = bce_loss.cuda()
        sigmoid = sigmoid.cuda()
    malconv.eval()
    embed = malconv.embed

    fixed_enmbed_tabel = embed(torch.arange(1, 257))

    """
    Attack
    """
    #   //
    add_data = np.random.randint(1, 257, adv_length)

    sccuce_count = 0
    total_count = 0
    write_add_data = []
    starttime = time.time()
    for step, batch_data in enumerate(validloader):
        exe_input = batch_data[0].cuda() if use_gpu else batch_data[0]
        data = exe_input[0].cpu().numpy()
        length = batch_data[2].item()
        label = batch_data[1].cuda() if use_gpu else batch_data[1]
        label = Variable(label.float(), requires_grad=False)
        if label.item() == 0 or length > first_n_byte - adv_length:
            continue
        total_count = total_count + 1

        insert_index = length
        adv_data, label = data2adv(
            malconv, add_data, data, insert_index, adv_length, use_gpu
        )
        mydata = adv_data[:]  #

        exe_input = num2tensor(adv_data, use_gpu, "long")
        embed_input = embed(exe_input)  #

        exe_input = num2tensor(adv_data, use_gpu, "long")
        pred = malconv(exe_input)
        prob = sigmoid(pred).cpu().data.numpy()[0][0]
        print("====================== ：", step + 1, "  ========================")
        print("prob: ", prob)

        continue_flag = False
        temp_count = [1]
        last_grad = 0
        new_data = mydata[:]
        for i in range(0, 30):
            if continue_flag or len(temp_count) == 0:
                continue
            #
            loss = bce_loss(pred, label)
            loss.backward()
            test_grad = malconv.embed_x.grad.data
            new_gard = test_grad + last_grad
            # new_gard = test_grad
            last_grad = test_grad

            #
            embed_sign = torch.sign(new_gard)
            adv_embed_sign = select_tensor(
                embed_sign, 1, insert_index, insert_index + adv_length
            )
            adv_embed_data = select_tensor(
                embed_input, 1, insert_index, insert_index + adv_length
            )
            if label.item() == 1:
                new_adv_embed_data = adv_embed_data + adv_step * adv_embed_sign
            else:
                new_adv_embed_data = adv_embed_data - adv_step * adv_embed_sign

            #
            temp_count = []
            for j in range(insert_index, insert_index + adv_length):
                adv_sign = embed_sign[0][j][:]
                zero_sign = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                if torch.eq(adv_sign, zero_sign).sum().item() != 8:
                    temp_count.append(j)

            if len(temp_count) != 0:

                select_data0 = select_tensor(embed_input, 1, begin=0, end=insert_index)
                select_data2 = select_tensor(
                    embed_input, 1, insert_index + adv_length, first_n_byte
                )
                new_embed_input = torch.cat(
                    (select_data0, new_adv_embed_data, select_data2), dim=1
                )

                new_data = embed2data(
                    mydata, temp_count, new_embed_input, fixed_enmbed_tabel, use_gpu
                )
            mydata = new_data[:]
            add_data = new_data[length : length + adv_length]
            exe_input = num2tensor(new_data, use_gpu, "long")
            embed_input = embed(exe_input)
            # malconv.eval()
            pred = malconv(exe_input)
            prob = sigmoid(pred).cpu().data.numpy()[0][0]
            print("Now is ", i, "  prob: ", prob)
            if prob < 0.5:
                print("prob<0.5,success.", "======:", i)
                continue_flag = True
                sccuce_count = sccuce_count + 1
                if write_flag == 1:
                    add_data2write = new_data[:]
                    write_add_data.append(add_data2write)

        if prob > 0.5:
            add_data2write = new_data[:]
            write_add_data.append(add_data2write)

    if write_flag == 1:
        write_add_data_array = np.array(write_add_data)
        np.save(npy_name + ".npy", write_add_data_array)

    endtime = time.time()
    print("=================：", sccuce_count / total_count)
    print("=================：", (endtime - starttime), "s,   adv_length", adv_length)


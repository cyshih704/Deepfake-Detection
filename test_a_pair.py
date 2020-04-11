    state_dict = torch.load('saved_model/FlowNet2.tar')['state_dict']
    model = models.FlowNet2(args).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    from dataloader.utils import image_transform
    from PIL import Image as pil_image
    import cv2
    import numpy as np
    from imageio import imwrite, imread
    img1 = cv2.resize(cv2.cvtColor(cv2.imread('1.png'), cv2.COLOR_BGR2RGB), (256, 256)) # h, w, c
    img2 = cv2.resize(cv2.cvtColor(cv2.imread('2.png'), cv2.COLOR_BGR2RGB), (256, 256))
    #img1 = imread('1.png')[:256, :256, :]
    #img2 = imread('2.png')[:256, :256, :]
    inputs = [img1, img2] # 2, h, w, c
    inputs = np.array(inputs).transpose(3, 0, 1, 2) # c, 2, h, w
    inputs = torch.from_numpy(inputs.astype(np.float32)).unsqueeze(0).to(DEVICE)
    
    """img1 = imread('1.png') # h, w, c
    img2 = imread('2.png')
    img1 = image_transform()(pil_image.fromarray(img1)).unsqueeze(0)*256 # 1, c, h, w
    img2 = image_transform()(pil_image.fromarray(img2)).unsqueeze(0)*256

    inputs = torch.cat((img1, img2), dim=0).unsqueeze(0).to(DEVICE).permute(0, 2, 1, 3, 4)"""
    



    result = model(inputs).squeeze()

    from flow_utils import writeFlow, flow2img


    data = result.data.cpu().numpy().transpose(1, 2, 0)
    data = flow2img(data)
    imwrite("flow1.png", data)
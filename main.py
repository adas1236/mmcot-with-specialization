import torch

def main():
    print("Hello from mmcot-with-specialization!")
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))

if __name__ == "__main__":
    main()

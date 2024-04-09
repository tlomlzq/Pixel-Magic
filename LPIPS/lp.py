import argparse
import os
import lpips

def lp(d0, d1):
	loss_fn = lpips.LPIPS(net='alex', version=0.1)
	loss_fn.cuda()

	files = os.listdir(d0)
	lpips_distances = []

	for file in files:
		if os.path.exists(os.path.join(d1, file)):
			# Load images
			img0 = lpips.im2tensor(lpips.load_image(os.path.join(d0, file)))  # RGB image from [-1,1]
			img1 = lpips.im2tensor(lpips.load_image(os.path.join(d1, file)))
			img0 = img0.cuda()
			img1 = img1.cuda()
			dist01 = loss_fn.forward(img0, img1)
			#print('%s: %.3f' % (file, dist01))
			#lpips_distances.append(dist01)
			lpips_distances.append(dist01.item())

	average_lpips = sum(lpips_distances) / len(lpips_distances)
	return round(average_lpips, 4)
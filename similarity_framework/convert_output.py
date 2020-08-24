sims_pre = []

with open('lpips_sim_waymo.txt') as f:
    data = f.read()
    sims_pre = data.split('(')[1:]
    
sims_post = []
for sim in sims_pre:
    sim = sim.split('): ')
    img1 = sim[0].split(', ')[0]
    img2 = sim[0].split(', ')[1]
    val = sim[1]
    sims_post.append(img1+','+img2+','+val)

with open('lpips_sim_waymo_processed.txt', 'w+') as o:
    for sim in sims_post:
        o.write(sim+'\n')

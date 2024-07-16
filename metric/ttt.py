import matplotlib.pyplot as plt

# 定义两个字典
dict1 = {
    0: 0.028845936059951782, 1: 0.008860809728503227, 2: 0.0053944033570587635, 3: 0.004114940296858549,
    4: 0.003790952730923891, 5: 0.003622218733653426, 7: 0.003443850902840495, 8: 0.003251237329095602,
    9: 0.00324839586392045, 10: 0.0030589245725423098, 11: 0.0029978875536471605, 14: 0.0029402109794318676,
    15: 0.002846060087904334, 16: 0.00278695416636765, 19: 0.0026950729079544544, 21: 0.0026302633341401815,
    23: 0.0026251303497701883, 24: 0.002604919020086527, 26: 0.002562996232882142, 27: 0.0025171348825097084,
    28: 0.0024996306747198105, 30: 0.0024867812171578407, 31: 0.0024577246513217688, 32: 0.0024501709267497063,
    34: 0.0023932410404086113, 37: 0.0023711889516562223, 41: 0.002368573099374771, 42: 0.0023677852004766464,
    44: 0.002277570543810725, 48: 0.0022678652312606573, 50: 0.0022585608530789614, 54: 0.002233222359791398,
    55: 0.0022233473137021065, 68: 0.0021916322875767946
}

dict2 = {
    0: 0.018428992480039597, 1: 0.010712646879255772, 2: 0.006527239456772804, 3: 0.005186718888580799,
    4: 0.004212115425616503, 5: 0.003990412689745426, 6: 0.003963611554354429, 7: 0.0037978803738951683,
    8: 0.003501079510897398, 9: 0.0034004561603069305, 11: 0.0033062950242310762, 12: 0.003171683521941304,
    13: 0.003146030008792877, 14: 0.003124753711745143, 15: 0.0030805759597569704, 16: 0.0030710964929312468,
    17: 0.002881405409425497, 19: 0.002859443658962846, 20: 0.0028373056557029486, 21: 0.002787020755931735,
    26: 0.0027551327366381884, 28: 0.002729313913732767, 30: 0.0027107649948447943, 31: 0.0027043030131608248,
    35: 0.0026971495244652033, 37: 0.002684610430151224, 38: 0.002665257314220071, 39: 0.002623707754537463,
    40: 0.0025947201065719128, 43: 0.002588958479464054, 48: 0.0025827588979154825, 49: 0.002572960453107953,
    51: 0.0025611973833292723, 57: 0.0025429276283830404, 65: 0.0025394896510988474, 67: 0.0025318728294223547,
    68: 0.002503311727195978, 81: 0.0024981466121971607, 86: 0.0024918513372540474, 88: 0.0024613484274595976,
    93: 0.0024340269155800343, 94: 0.0024180265609174967, 95: 0.0024054907262325287
}

dict3 = {0: 0.04486018046736717, 1: 0.0434659905731678, 2: 0.012731683440506458, 3: 0.009051883593201637,
         4: 0.006209848448634148, 5: 0.005414440296590328, 6: 0.004909833427518606, 8: 0.004727331455796957,
         10: 0.004656909499317408, 11: 0.004619542043656111, 13: 0.004556016530841589, 14: 0.004201309755444527,
         15: 0.004066524561494589, 19: 0.003966983407735825, 21: 0.0037595173344016075, 27: 0.003584436373785138,
         32: 0.003560988698154688, 37: 0.003456294070929289, 40: 0.0033204343635588884, 44: 0.0032908881548792124,
         48: 0.0032161215785890818, 51: 0.003093031235039234, 60: 0.0030013257637619972, 68: 0.0029578679241240025,
         74: 0.0029286949429661036, 79: 0.0029230001382529736, 81: 0.002890631090849638, 88: 0.002785695716738701}


dict4 = {0: 0.04347653314471245, 1: 0.019952598959207535, 2: 0.012452498078346252, 3: 0.008959423750638962,
         4: 0.0065176417119801044, 5: 0.005959082394838333, 6: 0.005557980854064226, 8: 0.004700041841715574,
         9: 0.004617694299668074, 10: 0.004429141525179148, 12: 0.004306104499846697, 13: 0.004227834288030863,
         15: 0.00422512274235487, 18: 0.004054172895848751, 19: 0.003984655253589153, 20: 0.003937730100005865,
         21: 0.003883805824443698, 22: 0.003776413854211569, 27: 0.0035142775159329176, 29: 0.003416403429582715,
         35: 0.0033997585996985435, 36: 0.003148126881569624, 37: 0.0031224226113408804, 38: 0.0030955702532082796,
         45: 0.003067001234740019, 46: 0.0030498511623591185, 51: 0.0030411756597459316, 54: 0.002967340871691704,
         60: 0.002932777628302574, 62: 0.0028933670837432146, 70: 0.0028185907285660505, 71: 0.0028082644566893578,
         78: 0.002746621146798134, 80: 0.002732616849243641, 82: 0.002700912533327937, 87: 0.0026565559674054384,
         92: 0.0026298672892153263, 95: 0.002543210983276367, 97: 0.0025380770675837994, 98: 0.0025030302349478006,
         106: 0.0024982488248497248, 112: 0.0024378907401114702, 113: 0.002393897157162428, 147: 0.002383463317528367}

dict5 = {0: 0.04347439110279083, 1: 0.015465322881937027, 2: 0.01065863762050867, 3: 0.007306898478418589,
         4: 0.005704944487661123, 5: 0.005387664306908846, 6: 0.004997424315661192, 9: 0.004450011067092419,
         14: 0.004339185077697039, 15: 0.0042128171771764755, 17: 0.004124893341213465, 18: 0.0038016566541045904,
         22: 0.0036018006503582, 23: 0.0035336629953235388, 26: 0.0034083605278283358, 27: 0.0033961967565119267,
         30: 0.003234999952837825, 34: 0.003150640521198511, 40: 0.0031387179624289274, 42: 0.0030975565314292908,
         43: 0.0029623941518366337, 56: 0.0029157104436308146, 61: 0.0028726004529744387, 64: 0.0028602206148207188,
         69: 0.002848708536475897, 73: 0.0028005968779325485, 81: 0.0027825504075735807, 82: 0.0027175843715667725,
         89: 0.002697253366932273, 96: 0.002686722669750452, 107: 0.002652216237038374, 111: 0.002623911714181304,
         112: 0.0026086694560945034, 118: 0.002606320194900036, 129: 0.002591583179309964, 130: 0.002555237151682377,
         132: 0.0025545686949044466, 139: 0.0025178685318678617, 142: 0.002492352854460478, 143: 0.002479757647961378,
         146: 0.0024638287723064423}

dict6 = {0: 0.04383465647697449, 1: 0.04279496893286705, 2: 0.04211781546473503, 3: 0.042084693908691406,
         4: 0.04108951985836029, 6: 0.039960119873285294, 7: 0.036576058715581894, 8: 0.03104894608259201,
         9: 0.02589857019484043, 10: 0.02409425377845764, 11: 0.021777089685201645, 12: 0.016878444701433182,
         14: 0.015056335367262363, 15: 0.01306360773742199, 18: 0.011673334054648876, 19: 0.01062680408358574,
         20: 0.010429622605443, 21: 0.009653393179178238, 22: 0.009337319992482662, 24: 0.009245618246495724,
         25: 0.008528080768883228, 27: 0.00817217119038105, 30: 0.00807622354477644, 31: 0.007652394473552704,
         33: 0.007158300373703241, 34: 0.006817334331572056, 36: 0.005913604516535997, 37: 0.005551581270992756,
         38: 0.005218775011599064, 39: 0.0048200832679867744, 41: 0.004693692550063133, 42: 0.004659972619265318,
         43: 0.004357202909886837, 44: 0.004298499319702387, 45: 0.004117471631616354, 47: 0.004089632537215948,
         48: 0.004077916499227285, 53: 0.004009881056845188, 54: 0.003976787906140089, 56: 0.003902943106368184,
         58: 0.003814921248704195, 61: 0.0037762608844786882, 64: 0.003717120038345456, 66: 0.003709173295646906,
         68: 0.0036977878771722317, 71: 0.003589208237826824, 76: 0.003510687965899706, 87: 0.0034584570676088333,
         89: 0.003404082963243127, 99: 0.0032848562113940716}

dict7 = {0: 0.3219362795352936, 1: 0.04886655882000923, 2: 0.01214693859219551, 3: 0.00868209172040224,
         4: 0.007845079526305199, 6: 0.004792474210262299, 7: 0.004428550601005554, 8: 0.004427690524607897,
         9: 0.004109567496925592, 11: 0.0037501670885831118, 12: 0.003725144313648343, 14: 0.0034962433855980635,
         16: 0.0034467962104827166, 19: 0.003330810694023967, 23: 0.0032271796371787786, 25: 0.003160614985972643,
         26: 0.003055223962292075, 28: 0.0030050978530198336, 29: 0.002999601187184453, 32: 0.002929247450083494,
         34: 0.002867882838472724, 36: 0.002851719269528985, 39: 0.002785745542496443, 45: 0.0027163871563971043,
         52: 0.002694737398996949, 54: 0.002663015853613615, 57: 0.0026365614030510187, 66: 0.0026357199531048536,
         69: 0.002620227402076125, 72: 0.0025771798100322485, 86: 0.0025350265204906464}

dict8 = {0: 0.020091362297534943, 1: 0.011977260001003742, 2: 0.00861852802336216, 3: 0.0066813817247748375,
         4: 0.005754328798502684, 5: 0.005372846964746714, 6: 0.004814942833036184, 7: 0.004542286042124033,
         8: 0.004346619360148907, 9: 0.004145571496337652, 10: 0.004065494053065777, 11: 0.003940538503229618,
         12: 0.003842331003397703, 13: 0.0038005616515874863, 14: 0.0037324419245123863, 15: 0.0036903417203575373,
         16: 0.003637430490925908, 17: 0.0035663156304508448, 19: 0.003549241693690419, 20: 0.0035104763228446245,
         23: 0.0033994484692811966, 25: 0.0033696249593049288, 26: 0.003357469104230404, 29: 0.0033021012786775827,
         30: 0.003260136814787984, 31: 0.0032521262764930725, 37: 0.0032019992358982563, 40: 0.003191382624208927,
         43: 0.0031685566063970327, 49: 0.003168092342093587}


dict9 = {0: 0.059245381504297256, 1: 0.011344671249389648, 2: 0.009807346388697624,
         3: 0.008084927685558796, 4: 0.006289259996265173, 5: 0.004466074053198099,
         6: 0.003936885390430689, 7: 0.0036499856505542994, 9: 0.0035598173271864653,
         10: 0.003458169288933277, 11: 0.0032821432687342167, 12: 0.003252657363191247,
         13: 0.0032415902242064476, 14: 0.003127498086541891, 15: 0.003087618388235569,
         16: 0.0030784422997385263, 18: 0.0030455542728304863, 19: 0.0028852205723524094,
         21: 0.002839484950527549, 24: 0.0026692915707826614, 27: 0.0025442650076001883,
         31: 0.0024820321705192327, 34: 0.0024114795960485935, 36: 0.002382977632805705,
         37: 0.0023315732832998037, 40: 0.002294622827321291, 44: 0.0022806234192103148,
         48: 0.002236384665593505, 51: 0.0022094908636063337, 57: 0.0021934364922344685,
         58: 0.0021657180041074753, 74: 0.002147224498912692, 75: 0.0021459839772433043,
         98: 0.002135221380740404}

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(list(dict1.keys()), list(dict1.values()), marker='o', label='Seq + Graph')
plt.plot(list(dict2.keys()), list(dict2.values()), marker='x', label='Seq-VAE + Graph-VAE')
# plt.plot(list(dict3.keys()), list(dict3.values()), marker='v', label='Seq-VQVAE')
# plt.plot(list(dict4.keys()), list(dict4.values()), marker='*', label='Seq-VQVAE + Graph')
# plt.plot(list(dict5.keys()), list(dict5.values()), marker='+', label='Seq-VQVAE + Graph-VAE')
plt.plot(list(dict6.keys()), list(dict6.values()), marker='p', label='TD-VAE')
plt.plot(list(dict7.keys()), list(dict7.values()), marker='s', label='Transformer-Mamba + Graph')
plt.plot(list(dict8.keys()), list(dict8.values()), marker='^', label='Mamba + Graph')
plt.plot(list(dict9.keys()), list(dict9.values()), marker='*', label='Transformer-MoE + Graph')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('RepCon vs RepCon-VAE')
plt.legend()
plt.grid(True)
plt.show()


# import dgl.data
#
# dataset = dgl.data.CoraGraphDataset()
# print('Number of categories:', dataset.num_classes)

# import torch
# print(torch.__version__)
#
# x = torch.randn((1,2,3))
# x = x.to('cuda')
# print(x)


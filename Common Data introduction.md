# 1.PA-100K

Images: 100000 (Training:80000, Validation: 10000, testing: 10000)

Attributes labels: 26

'Female'

'AgeOver60'

'Age18-60'

'AgeLess18'

'Front'

'Side'

'Back'

'Hat'

'Glasses'

'HandBag'

'ShoulderBag'

'Backpack'

'HoldObjectsInFront'

'ShortSleeve'

'LongSleeve'

'UpperStride'

'UpperLogo'

'UpperPlaid'

'UpperSplice'

'LowerStripe'

'LowerPattern'

'LongCoat'

'Trousers'

'Shorts'

'Skirt&Dress'

'boots'

Image resolution: low

mean size: Width: 64 Height: 194

# 2.Parse27K

Images: 40755(Training:27482, Validation: 6618,  testing: 6655)

Attributes labels: 12

Attributes:

Occlusion ----0,1

Gender(none, male, female)-------1,2,3

Orientation(none, front, back, left, right,)----1,2,3,4,5

Posture(none, walking/ running ,standing, sitting)----1,2,3,4

Handbag on shoulder left(none, yes,no)—1,2,3

Handbag on shoulder right

Has bag in hand right

Has bag in hand left

Has backpack

Has trolley

Is pushing 

Is talking

Image resolution: low

mean size: Width: 128 Height: 192

# 3.PETA

Images: 19000 

Image resolution: median

Attributes labels: 61

Attribute:

Age16-30 

Age31-45

Age46-60

AgeAbove61

Backpack

CarryingOther

Casual lower

Casual upper

Formal lower

Formal upper

Hat

Jacket

Jeans

Leather shoes

Logo

Long hair 

Male

MessengerBag

Muffler

No accessory

No carrying

Plaid

Plastic bag

Sandals

Shoes

Shorts

ShortSleeve

Skirt

Sneaker

Stripes

Sunglasses

Trousers

T-shirt

UpperOther

V-Neck
...

![PETA](http://mmlab.ie.cuhk.edu.hk/projects/PETA_files/pie.png)


Datasets	#Images	Camera angle	View point	Illumination	Resolution	Scene

3DPeS	1012	high	varying	varying	from 31x100 to 236 x 178	outdoor

CAVIAR4REID	1220	ground	varying	low	from 17x39 to 72x141	outdoor

CUHK	4563	high	varying	varying	80x160	indoor

GRID	1275	varying	frontal & back	low	from 29x67 to 169x365	indoor

i-LIDS	477	medium	back	high	from 32x76 to 115x294	outdoor

MIT	888	ground	back	high	64x128	outdoor

PRID	1134	high	profile	low	64x128	outdoor

SARC3D	200	medium	varying	varying	from 54x187 to 150x307	outdoor

TownCentre	6967	medium	varying	medium	from 44x109 to 148x332	outdoor

VIPeR	1264	ground	varying	varying	48x128	outdoor

size ranging: 17-by-39 to 169-by-365


# 4.RAP

Images: 41585 

Attributes labels: 92
'Female'
'AgeLess16'
'Age17-30'
'Age31-45'
'BodyFat'
'BodyNormal'
'BodyThin'
'Customer'
'Clerk'
'hs-BaldHead'
'hs-LongHair'
'hs-BlackHair'
'hs-Hat'
'hs-Glasses'
'hs-Muffler'
'ub-Shirt'
'ub-Sweater'
'ub-Vest'
'ub-TShirt'
'ub-Cotton'
'ub-Jacket'
'ub-SuitUp'
'ub-Tight'
'ub-ShortSleeve'
'lb-LongTrousers'
'lb-Skirt'
'lb-ShortSkirt'
'lb-Dress'
'lb-Jeans'
'lb-TightTrousers'
'shoes-Leather'
'shoes-Sport'
'shoes-Boots'
'shoes-Cloth'
'shoes-Casual'
'attach-Backpack'
'attach-SingleShoulderBag'
'attach-HandBag'
'attach-Box'
'attach-PlasticBag'
'attach-PaperBag'
'attach-HandTrunk'
'attach-Other'
'action-Calling'
'action-Talking'
'action-Gathering'
'action-Holding'
'action-Pusing'
'action-Pulling'
'action-CarrybyArm'
'action-CarrybyHand'
'faceFront'
'faceBack'
'faceLeft'
'faceRight'
'occlusionLeft'
'occlusionRight'
'occlusionUp'
'occlusionDown'
'occlusion-Environment'
'occlusion-Attachment'
'occlusion-Person'
'occlusion-Other'
'up-Black'
'up-White'
'up-Gray'
'up-Red'
'up-Green'
'up-Blue'
'up-Yellow'
'up-Brown'
'up-Purple'
'up-Pink'
'up-Orange'
'up-Mixture'
'low-Black'
'low-White'
'low-Gray'
'low-Red'
'low-Green'
'low-Blue'
'low-Yellow'
'low-Mixture'
'shoes-Black'
'shoes-White'
'shoes-Gray'
'shoes-Red'
'shoes-Green'
'shoes-Blue'
'shoes-Yellow'
'shoes-Brown'
'shoes-Mixture'

Special: 

RAP_annotation.position is the absolute coordinate of the person bounding box of fullbody, head-shoulder, upperbody, lowerbody in the full image.
Each of bounding box has four points, including (x,y,w,h). 




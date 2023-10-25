labels_off = """Alarm_Clock  Chair       File_Cabinet  Knives      Pan           Scissors     ToothBrush
Backpack     Clipboards  Flipflops     Lamp_Shade  Paper_Clip    Screwdriver  Toys
Batteries    Computer    Flowers       Laptop      Pen           Shelf        Trash_Can
Bed          Couch       Folder        Marker      Pencil        Sink         TV
Bike         Curtains    Fork          Monitor     Postit_Notes  Sneakers     Webcam
Bottle       Desk_Lamp   Glasses       Mop         Printer       Soda
Bucket       Drill       Hammer        Mouse       Push_Pin      Speaker
Calculator   Eraser      Helmet        Mug         Radio         Spoon
Calendar     Exit_Sign   Kettle        Notebook    Refrigerator  Table
Candles      Fan         Keyboard      Oven        Ruler         Telephone"""
labels_domain = """aircraft_carrier  chandelier     harp             palm_tree       spider
airplane          church         hat              panda           spoon
alarm_clock       circle         headphones       pants           spreadsheet
ambulance         clarinet       hedgehog         paper_clip      square
angel             clock          helicopter       parachute       squiggle
animal_migration  cloud          helmet           parrot          squirrel
ant               coffee_cup     hexagon          passport        stairs
anvil             compass        hockey_puck      peanut          star
apple             computer       hockey_stick     pear            steak
arm               cookie         horse            peas            stereo
asparagus         cooler         hospital         pencil          stethoscope
axe               couch          hot_air_balloon  penguin         stitches
backpack          cow            hot_dog          piano           stop_sign
banana            crab           hot_tub          pickup_truck    stove
bandage           crayon         hourglass        picture_frame   strawberry
barn              crocodile      house            pig             streetlight
baseball          crown          house_plant      pillow          string_bean
baseball_bat      cruise_ship    hurricane        pineapple       submarine
basket            cup            ice_cream        pizza           suitcase
basketball        diamond        jacket           pliers          sun
bat               dishwasher     jail             police_car      swan
bathtub           diving_board   kangaroo         pond            sweater
beach             dog            key              pool            swing_set
bear              dolphin        keyboard         popsicle        sword
beard             donut          knee             postcard        syringe
bed               door           knife            potato          table
bee               dragon         ladder           power_outlet    teapot
belt              dresser        lantern          purse           teddy-bear
bench             drill          laptop           rabbit          telephone
bicycle           drums          leaf             raccoon         television
binoculars        duck           leg              radio           tennis_racquet
bird              dumbbell       light_bulb       rain            tent
birthday_cake     ear            lighter          rainbow         The_Eiffel_Tower
blackberry        elbow          lighthouse       rake            The_Great_Wall_of_China
blueberry         elephant       lightning        remote_control  The_Mona_Lisa
book              envelope       line             rhinoceros      tiger
boomerang         eraser         lion             rifle           toaster
bottlecap         eye            lipstick         river           toe
bowtie            eyeglasses     lobster          roller_coaster  toilet
bracelet          face           lollipop         rollerskates    tooth
brain             fan            mailbox          sailboat        toothbrush
bread             feather        map              sandwich        toothpaste
bridge            fence          marker           saw             tornado
broccoli          finger         matches          saxophone       tractor
broom             fire_hydrant   megaphone        school_bus      traffic_light
bucket            fireplace      mermaid          scissors        train
bulldozer         firetruck      microphone       scorpion        tree
bus               fish           microwave        screwdriver     triangle
bush              flamingo       monkey           sea_turtle      trombone
butterfly         flashlight     moon             see_saw         truck
cactus            flip_flops     mosquito         shark           trumpet
cake              floor_lamp     motorbike        sheep           t-shirt
calculator        flower         mountain         shoe            umbrella
calendar          flying_saucer  mouse            shorts          underwear
camel             foot           moustache        shovel          van
camera            fork           mouth            sink            vase
camouflage        frog           mug              skateboard      violin
campfire          frying_pan     mushroom         skull           washing_machine
candle            garden         nail             skyscraper      watermelon
cannon            garden_hose    necklace         sleeping_bag    waterslide
canoe             giraffe        nose             smiley_face     whale
car               goatee         ocean            snail           wheel
carrot            golf_club      octagon          snake           windmill
castle            grapes         octopus          snorkel         wine_bottle
cat               grass          onion            snowflake       wine_glass
ceiling_fan       guitar         oven             snowman         wristwatch
cello             hamburger      owl              soccer_ball     yoga
cell_phone        hammer         paintbrush       sock            zebra
chair             hand           paint_can        speedboat       zigzag""" 
labels_ = [i for i in labels_domain.split(" ") if i != ""]
print(labels_)
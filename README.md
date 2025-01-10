# House Price Prediction 

Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veriseti kullanılarak,
farklı tipteki evlerin fiyatlarına ilişkin bir makine öğrenmesi projesi gerçekleştirilmek istenmektedir.

Ames, Lowa’daki konut evlerinden oluşan bu veri seti içerisinde 79 açıklayıcı değişken bulunduruyor.
Kaggle üzerinde bir yarışması da bulunan projenin veri seti ve yarışma sayfasına aşağıdaki linkten ulaşabilirsiniz.
Veri seti bir kaggle yarışmasına ait olduğundan dolayı train ve test olmak üzere iki farklı csv dosyası vardır.
Test veri setinde ev fiyatları boş bırakılmış olup, bu değerleri sizin tahmin etmeniz beklenmektedir.

Degiskenler
1. Id
2. MSSubClass: Identifies the type of dwelling involved in the sale.
         20	1-STORY 1946 & NEWER ALL STYLES
         30	1-STORY 1945 & OLDER
         40	1-STORY W/FINISHED ATTIC ALL AGES
#         45	1-1/2 STORY - UNFINISHED ALL AGES
#         50	1-1/2 STORY FINISHED ALL AGES
#         60	2-STORY 1946 & NEWER
#         70	2-STORY 1945 & OLDER
#         75	2-1/2 STORY ALL AGES
#         80	SPLIT OR MULTI-LEVEL
#         85	SPLIT FOYER
#         90	DUPLEX - ALL STYLES AND AGES
#        120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
#        150	1-1/2 STORY PUD - ALL AGES
#        160	2-STORY PUD - 1946 & NEWER
#        180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
#        190	2 FAMILY CONVERSION - ALL STYLES AND AGES
# 3. MSZoning: Identifies the general zoning classification of the sale.
#        A	Agriculture
#        C	Commercial
#        FV	Floating Village Residential
#        I	Industrial
#        RH	Residential High Density
#        RL	Residential Low Density
#        RP	Residential Low Density Park
#        RM	Residential Medium Density
# 4. LotFrontage: Linear feet of street connected to property
# 5. LotArea: Lot size in square feet
# 6. Street: Type of road access to property
#        Grvl	Gravel
#        Pave	Paved
# 7. Alley: Type of alley access to property
#        Grvl	Gravel
#        Pave	Paved
#        NA 	No alley access
# 8. LotShape: General shape of property
#        Reg	Regular
#        IR1	Slightly irregular
#        IR2	Moderately Irregular
#        IR3	Irregular
# 9. LandContour: Flatness of the property
#        Lvl	Near Flat/Level
#        Bnk	Banked - Quick and significant rise from street grade to building
#        HLS	Hillside - Significant slope from side to side
#        Low	Depression
# 10. Utilities: Type of utilities available
#        AllPub	All public Utilities (E,G,W,& S)
#        NoSewr	Electricity, Gas, and Water (Septic Tank)
#        NoSeWa	Electricity and Gas Only
#        ELO	Electricity only
# 11. LotConfig: Lot configuration
#        Inside	Inside lot
#        Corner	Corner lot
#        CulDSac	Cul-de-sac
#        FR2	Frontage on 2 sides of property
#        FR3	Frontage on 3 sides of property
# 12. LandSlope: Slope of property
#        Gtl	Gentle slope
#        Mod	Moderate Slope
#        Sev	Severe Slope
# 13. Neighborhood: Physical locations within Ames city limits
#        Blmngtn	Bloomington Heights
#        Blueste	Bluestem
#        BrDale	    Briardale
#        BrkSide	Brookside
#        ClearCr	Clear Creek
#        CollgCr	College Creek
#        Crawfor	Crawford
#        Edwards	Edwards
#        Gilbert	Gilbert
#        IDOTRR	    Iowa DOT and Rail Road
#        MeadowV	Meadow Village
#        Mitchel	Mitchell
#        Names	    North Ames
#        NoRidge	Northridge
#        NPkVill	Northpark Villa
#        NridgHt	Northridge Heights
#        NWAmes	    Northwest Ames
#        OldTown	Old Town
#        SWISU	    South & West of Iowa State University
#        Sawyer	    Sawyer
#        SawyerW	Sawyer West
#        Somerst	Somerset
#        StoneBr	Stone Brook
#        Timber	    Timberland
#        Veenker	Veenker
# 14. Condition1: Proximity to various conditions
#        Artery	Adjacent to arterial street
#        Feedr	Adjacent to feeder street
#        Norm	Normal
#        RRNn	Within 200' of North-South Railroad
#        RRAn	Adjacent to North-South Railroad
#        PosN	Near positive off-site feature--park, greenbelt, etc.
#        PosA	Adjacent to postive off-site feature
#        RRNe	Within 200' of East-West Railroad
#        RRAe	Adjacent to East-West Railroad
# 15. Condition2: Proximity to various conditions (if more than one is present)
#        Artery	Adjacent to arterial street
#        Feedr	Adjacent to feeder street
#        Norm	Normal
#        RRNn	Within 200' of North-South Railroad
#        RRAn	Adjacent to North-South Railroad
#        PosN	Near positive off-site feature--park, greenbelt, etc.
#        PosA	Adjacent to postive off-site feature
#        RRNe	Within 200' of East-West Railroad
#        RRAe	Adjacent to East-West Railroad
# 16. BldgType: Type of dwelling
#        1Fam	Single-family Detached
#        2FmCon	Two-family Conversion; originally built as one-family dwelling
#        Duplx	Duplex
#        TwnhsE	Townhouse End Unit
#        TwnhsI	Townhouse Inside Unit
# 17. HouseStyle: Style of dwelling
#        1Story	One story
#        1.5Fin	One and one-half story: 2nd level finished
#        1.5Unf	One and one-half story: 2nd level unfinished
#        2Story	Two story
#        2.5Fin	Two and one-half story: 2nd level finished
#        2.5Unf	Two and one-half story: 2nd level unfinished
#        SFoyer	Split Foyer
#        SLvl	Split Level
# 18. OverallQual: Rates the overall material and finish of the house
#        10	Very Excellent
#        9	Excellent
#        8	Very Good
#        7	Good
#        6	Above Average
#        5	Average
#        4	Below Average
#        3	Fair
#        2	Poor
#        1	Very Poor
# 19. OverallCond: Rates the overall condition of the house
#        10	Very Excellent
#        9	Excellent
#        8	Very Good
#        7	Good
#        6	Above Average
#        5	Average
#        4	Below Average
#        3	Fair
#        2	Poor
#        1	Very Poor
# 20. YearBuilt: Original construction date
# 21. YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
# 22. RoofStyle: Type of roof
#        Flat	Flat
#        Gable	Gable
#        Gambrel	Gabrel (Barn)
#        Hip	Hip
#        Mansard	Mansard
#        Shed	Shed
# 23. RoofMatl: Roof material
#        ClyTile	Clay or Tile
#        CompShg	Standard (Composite) Shingle
#        Membran	Membrane
#        Metal	Metal
#        Roll	Roll
#        Tar&Grv	Gravel & Tar
#        WdShake	Wood Shakes
#        WdShngl	Wood Shingles
# 24. Exterior1st: Exterior covering on house
#        AsbShng	Asbestos Shingles
#        AsphShn	Asphalt Shingles
#        BrkComm	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        CemntBd	Cement Board
#        HdBoard	Hard Board
#        ImStucc	Imitation Stucco
#        MetalSd	Metal Siding
#        Other	Other
#        Plywood	Plywood
#        PreCast	PreCast
#        Stone	Stone
#        Stucco	Stucco
#        VinylSd	Vinyl Siding
#        Wd Sdng	Wood Siding
#        WdShing	Wood Shingles
# 25. Exterior2nd: Exterior covering on house (if more than one material)
#        AsbShng	Asbestos Shingles
#        AsphShn	Asphalt Shingles
#        BrkComm	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        CemntBd	Cement Board
#        HdBoard	Hard Board
#        ImStucc	Imitation Stucco
#        MetalSd	Metal Siding
#        Other	Other
#        Plywood	Plywood
#        PreCast	PreCast
#        Stone	Stone
#        Stucco	Stucco
#        VinylSd	Vinyl Siding
#        Wd Sdng	Wood Siding
#        WdShing	Wood Shingles
# 26. MasVnrType: Masonry veneer type
#        BrkCmn	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        None	None
#        Stone	Stone
# 27. MasVnrArea: Masonry veneer area in square feet
# 28. ExterQual: Evaluates the quality of the material on the exterior
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor
# 29. ExterCond: Evaluates the present condition of the material on the exterior
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor
# 30. Foundation: Type of foundation
#        BrkTil	Brick & Tile
#        CBlock	Cinder Block
#        PConc	Poured Contrete
#        Slab	Slab
#        Stone	Stone
#        Wood	Wood
# 31. BsmtQual: Evaluates the height of the basement
#        Ex	Excellent (100+ inches)
#        Gd	Good (90-99 inches)
#        TA	Typical (80-89 inches)
#        Fa	Fair (70-79 inches)
#        Po	Poor (<70 inches
#        NA	No Basement
# 32. BsmtCond: Evaluates the general condition of the basement
#        Ex	Excellent
#        Gd	Good
#        TA	Typical - slight dampness allowed
#        Fa	Fair - dampness or some cracking or settling
#        Po	Poor - Severe cracking, settling, or wetness
#        NA	No Basement
# 33. BsmtExposure: Refers to walkout or garden level walls
#        Gd	Good Exposure
#        Av	Average Exposure (split levels or foyers typically score average or above)
#        Mn	Mimimum Exposure
#        No	No Exposure
#        NA	No Basement
# 34. BsmtFinType1: Rating of basement finished area
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinshed
#        NA	No Basement
# 35. BsmtFinSF1: Type 1 finished square feet
# 36. BsmtFinType2: Rating of basement finished area (if multiple types)
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinshed
#        NA	No Basement
# 37. BsmtFinSF2: Type 2 finished square feet
# 38. BsmtUnfSF: Unfinished square feet of basement area
# 39. TotalBsmtSF: Total square feet of basement area
# 40. Heating: Type of heating
#        Floor	Floor Furnace
#        GasA	Gas forced warm air furnace
#        GasW	Gas hot water or steam heat
#        Grav	Gravity furnace
#        OthW	Hot water or steam heat other than gas
#        Wall	Wall furnace
# 41. HeatingQC: Heating quality and condition
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor
# 42. CentralAir: Central air conditioning
#        N	No
#        Y	Yes
# 43. Electrical: Electrical system
#        SBrkr	Standard Circuit Breakers & Romex
#        FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)
#        FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
#        FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
#        Mix	Mixed
# 44. 1stFlrSF: First Floor square feet
# 45. 2ndFlrSF: Second floor square feet
# 46. LowQualFinSF: Low quality finished square feet (all floors)
# 47. GrLivArea: Above grade (ground) living area square feet
# 48. BsmtFullBath: Basement full bathrooms
# 49. BsmtHalfBath: Basement half bathrooms
# 50. FullBath: Full bathrooms above grade
# 51. HalfBath: Half baths above grade
# 52. BedroomAbvGr: Bedrooms above grade (does NOT include basement bedrooms)
# 53. KitchenAbvGr: Kitchens above grade
# 54. KitchenQual: Kitchen quality
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
# 55. TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# 56. Functional: Home functionality (Assume typical unless deductions are warranted)
#        Typ	Typical Functionality
#        Min1	Minor Deductions 1
#        Min2	Minor Deductions 2
#        Mod	Moderate Deductions
#        Maj1	Major Deductions 1
#        Maj2	Major Deductions 2
#        Sev	Severely Damaged
#        Sal	Salvage only
# 57. Fireplaces: Number of fireplaces
# 58. FireplaceQu: Fireplace quality
#        Ex	Excellent - Exceptional Masonry Fireplace
#        Gd	Good - Masonry Fireplace in main level
#        TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
#        Fa	Fair - Prefabricated Fireplace in basement
#        Po	Poor - Ben Franklin Stove
#        NA	No Fireplace
# 59. GarageType: Garage location
#        2Types	More than one type of garage
#        Attchd	Attached to home
#        Basment	Basement Garage
#        BuiltIn	Built-In (Garage part of house - typically has room above garage)
#        CarPort	Car Port
#        Detchd	Detached from home
#        NA	No Garage
# 60. GarageYrBlt: Year garage was built
# 61. GarageFinish: Interior finish of the garage
#        Fin	Finished
#        RFn	Rough Finished
#        Unf	Unfinished
#        NA	No Garage
# 62. GarageCars: Size of garage in car capacity
# 63. GarageArea: Size of garage in square feet
# 64. GarageQual: Garage quality
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
# 65. GarageCond: Garage condition
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
# 66. PavedDrive: Paved driveway
#        Y	Paved
#        P	Partial Pavement
#        N	Dirt/Gravel
# 67. WoodDeckSF: Wood deck area in square feet
# 68. OpenPorchSF: Open porch area in square feet
# 69. EnclosedPorch: Enclosed porch area in square feet
# 70. 3SsnPorch: Three season porch area in square feet
# 71. ScreenPorch: Screen porch area in square feet
# 72. PoolArea: Pool area in square feet
# 73. PoolQC: Pool quality
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        NA	No Pool
# 74. Fence: Fence quality
#        GdPrv	Good Privacy
#        MnPrv	Minimum Privacy
#        GdWo	Good Wood
#        MnWw	Minimum Wood/Wire
#        NA	No Fence
# 75. MiscFeature: Miscellaneous feature not covered in other categories
#        Elev	Elevator
#        Gar2	2nd Garage (if not described in garage section)
#        Othr	Other
#        Shed	Shed (over 100 SF)
#        TenC	Tennis Court
#        NA	None
# 76. MiscVal: $Value of miscellaneous feature
# 77. MoSold: Month Sold (MM)
# 78. YrSold: Year Sold (YYYY)
# 79. SaleType: Type of sale
#        WD 	Warranty Deed - Conventional
#        CWD	Warranty Deed - Cash
#        VWD	Warranty Deed - VA Loan
#        New	Home just constructed and sold
#        COD	Court Officer Deed/Estate
#        Con	Contract 15% Down payment regular terms
#        ConLw	Contract Low Down payment and low interest
#        ConLI	Contract Low Interest
#        ConLD	Contract Low Down
#        Oth	Other
# 80. SaleCondition: Condition of sale
#        Normal	Normal Sale
#        Abnorml	Abnormal Sale -  trade, foreclosure, short sale
#        AdjLand	Adjoining Land Purchase
#        Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit
#        Family	Sale between family members
#        Partial	Home was not completed when last assessed (associated with New Homes)
# 81. SalePrice

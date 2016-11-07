-- Tier 1 on Google Bigquery doesn't allow for properties to be assigned after joining to the lake data (a one to many relationship)
-- Likely due to how the query optimizer handles the query
-- Limited data in CTEs before joining to the lakes to work around this
-- Additionally, fuzzy property types (e.x. a primarily agricultural property that is secondarily residential) are excluded to prevent counting multiple times
-- This would have been handled in a single case expression (as they evaluate sequentially), which was not possible with the hardware constraints


WITH lake AS (
SELECT
 LAKE_NAME AS LakeName
, DNR_ID_SITE_NUMBER
FROM `datadive-142319.mces_lakes.1999_2014_monitoring_data`
GROUP BY LAKE_NAME, DNR_ID_SITE_NUMBER
)

, residential AS (
SELECT
 USE1_DESC AS PropertyType
, centroid_long
, centroid_lat
FROM `datadive-142319.metrogis_parcels.2014_tax_parcel_data`
WHERE 

  -- Residential
  (LTRIM(LOWER(USE1_DESC)) LIKE '1__%'
  OR LOWER(USE1_DESC) LIKE '%residential%'
  OR LOWER(USE1_DESC) LIKE '%res%'
  OR LOWER(USE1_DESC) LIKE '%house%'
  OR LOWER(USE1_DESC) LIKE '%condo%'
  OR LOWER(USE1_DESC) LIKE '%apartment%'
  OR LOWER(USE1_DESC) LIKE '%apt%'
  OR LOWER(USE1_DESC) LIKE '%plex%'
  OR LOWER(USE1_DESC) LIKE '%bungalo%'
  OR LOWER(USE1_DESC) LIKE '%housing%'
  OR LOWER(USE1_DESC) LIKE '%home%'
  OR LOWER(USE1_DESC) LIKE '%family%')

  -- Agricultural
  AND LOWER(USE1_DESC) NOT LIKE '2__%'
  AND LOWER(USE1_DESC) NOT LIKE '%ag%'
  AND LOWER(USE1_DESC) NOT LIKE '%farm%'
  AND LOWER(USE1_DESC) NOT LIKE '%rural%'

  -- Commercial
  AND LOWER(USE1_DESC) NOT LIKE '3__%'
  AND LOWER(USE1_DESC) NOT LIKE '%commercial%'
  AND LOWER(USE1_DESC) NOT LIKE '%machinery%'
  AND LOWER(USE1_DESC) NOT LIKE '%recreational%'
  AND LOWER(USE1_DESC) NOT LIKE '%golf%'
  AND LOWER(USE1_DESC) NOT LIKE '%coop%'

  -- Industrial
  AND LOWER(USE1_DESC) NOT LIKE '%ind%'
  AND LOWER(USE1_DESC) != '305 industrial'

  -- Public
  AND LOWER(USE1_DESC) NOT LIKE '9__%'
  AND LOWER(USE1_DESC) NOT LIKE '%public%'
  AND LOWER(USE1_DESC) NOT LIKE '%muni%'
  AND LOWER(USE1_DESC) NOT LIKE '%rail%'
  AND LOWER(USE1_DESC) NOT LIKE '%church%'
  AND LOWER(USE1_DESC) NOT LIKE '%school%'
  AND LOWER(USE1_DESC) NOT LIKE '%forest%'
  AND LOWER(USE1_DESC) NOT LIKE '%state%'
  AND LOWER(USE1_DESC) NOT LIKE '%county%'
  AND LOWER(USE1_DESC) NOT LIKE '%util%'
  AND LOWER(USE1_DESC) NOT LIKE '%college%'
  AND LOWER(USE1_DESC) NOT LIKE '%cem%'
  AND LOWER(USE1_DESC) NOT LIKE '%common%'
  AND LOWER(USE1_DESC) NOT LIKE '%road%'
  AND LOWER(USE1_DESC) NOT LIKE '%fed%'
  AND LOWER(USE1_DESC) NOT LIKE '%tax%'
  AND LOWER(USE1_DESC) NOT LIKE '%dnr%'
  AND LOWER(USE1_DESC) NOT LIKE '%charit%'
  AND LOWER(USE1_DESC) NOT LIKE '%serv%'
  AND LOWER(USE1_DESC) NOT LIKE '%hosp%'
  AND LOWER(USE1_DESC) NOT LIKE '%park%'

GROUP BY USE1_DESC, centroid_long, centroid_lat
)

, agriculture AS (
SELECT
 USE1_DESC AS PropertyType
, centroid_long
, centroid_lat
FROM `datadive-142319.metrogis_parcels.2014_tax_parcel_data`
WHERE 

  -- Agricultural
   (LOWER(USE1_DESC) LIKE '2__%'
   OR LOWER(USE1_DESC) LIKE '%ag%'
   OR LOWER(USE1_DESC) LIKE '%farm%'
   OR LOWER(USE1_DESC) LIKE '%rural%')

  -- Residential
  AND LTRIM(LOWER(USE1_DESC)) NOT LIKE '1__%'
  AND LOWER(USE1_DESC) NOT LIKE '%residential%'
  AND LOWER(USE1_DESC) NOT LIKE '%res%'
  AND LOWER(USE1_DESC) NOT LIKE '%house%'
  AND LOWER(USE1_DESC) NOT LIKE '%condo%'
  AND LOWER(USE1_DESC) NOT LIKE '%apartment%'
  AND LOWER(USE1_DESC) NOT LIKE '%apt%'
  AND LOWER(USE1_DESC) NOT LIKE '%plex%'
  AND LOWER(USE1_DESC) NOT LIKE '%bungalo%'
  AND LOWER(USE1_DESC) NOT LIKE '%housing%'
  AND LOWER(USE1_DESC) NOT LIKE '%home%'
  AND LOWER(USE1_DESC) NOT LIKE '%family%'

  -- Commercial
  AND LOWER(USE1_DESC) NOT LIKE '3__%'
  AND LOWER(USE1_DESC) NOT LIKE '%commercial%'
  AND LOWER(USE1_DESC) NOT LIKE '%machinery%'
  AND LOWER(USE1_DESC) NOT LIKE '%recreational%'
  AND LOWER(USE1_DESC) NOT LIKE '%golf%'
  AND LOWER(USE1_DESC) NOT LIKE '%coop%'

  -- Industrial
  AND LOWER(USE1_DESC) NOT LIKE '%ind%'
  AND LOWER(USE1_DESC) != '305 industrial'

  -- Public
  AND LOWER(USE1_DESC) NOT LIKE '9__%'
  AND LOWER(USE1_DESC) NOT LIKE '%public%'
  AND LOWER(USE1_DESC) NOT LIKE '%muni%'
  AND LOWER(USE1_DESC) NOT LIKE '%rail%'
  AND LOWER(USE1_DESC) NOT LIKE '%church%'
  AND LOWER(USE1_DESC) NOT LIKE '%school%'
  AND LOWER(USE1_DESC) NOT LIKE '%forest%'
  AND LOWER(USE1_DESC) NOT LIKE '%state%'
  AND LOWER(USE1_DESC) NOT LIKE '%county%'
  AND LOWER(USE1_DESC) NOT LIKE '%util%'
  AND LOWER(USE1_DESC) NOT LIKE '%college%'
  AND LOWER(USE1_DESC) NOT LIKE '%cem%'
  AND LOWER(USE1_DESC) NOT LIKE '%common%'
  AND LOWER(USE1_DESC) NOT LIKE '%road%'
  AND LOWER(USE1_DESC) NOT LIKE '%fed%'
  AND LOWER(USE1_DESC) NOT LIKE '%tax%'
  AND LOWER(USE1_DESC) NOT LIKE '%dnr%'
  AND LOWER(USE1_DESC) NOT LIKE '%charit%'
  AND LOWER(USE1_DESC) NOT LIKE '%serv%'
  AND LOWER(USE1_DESC) NOT LIKE '%hosp%'
  AND LOWER(USE1_DESC) NOT LIKE '%park%'
)

, commercial AS (
SELECT
 USE1_DESC AS PropertyType
, centroid_long
, centroid_lat
FROM `datadive-142319.metrogis_parcels.2014_tax_parcel_data`
WHERE 

   -- Commercial
   (LOWER(USE1_DESC) LIKE '3__%'
   OR LOWER(USE1_DESC) LIKE '%commercial%'
   OR LOWER(USE1_DESC) LIKE '%machinery%'
   OR LOWER(USE1_DESC) LIKE '%recreational%'
   OR LOWER(USE1_DESC) LIKE '%golf%'
   OR LOWER(USE1_DESC) LIKE '%coop%')

  -- Residential
  AND LTRIM(LOWER(USE1_DESC)) NOT LIKE '1__%'
  AND LOWER(USE1_DESC) NOT LIKE '%residential%'
  AND LOWER(USE1_DESC) NOT LIKE '%res%'
  AND LOWER(USE1_DESC) NOT LIKE '%house%'
  AND LOWER(USE1_DESC) NOT LIKE '%condo%'
  AND LOWER(USE1_DESC) NOT LIKE '%apartment%'
  AND LOWER(USE1_DESC) NOT LIKE '%apt%'
  AND LOWER(USE1_DESC) NOT LIKE '%plex%'
  AND LOWER(USE1_DESC) NOT LIKE '%bungalo%'
  AND LOWER(USE1_DESC) NOT LIKE '%housing%'
  AND LOWER(USE1_DESC) NOT LIKE '%home%'
  AND LOWER(USE1_DESC) NOT LIKE '%family%'

  -- Industrial
  AND LOWER(USE1_DESC) NOT LIKE '%ind%'
  AND LOWER(USE1_DESC) != '305 industrial'

  -- Public
  AND LOWER(USE1_DESC) NOT LIKE '9__%'
  AND LOWER(USE1_DESC) NOT LIKE '%public%'
  AND LOWER(USE1_DESC) NOT LIKE '%muni%'
  AND LOWER(USE1_DESC) NOT LIKE '%rail%'
  AND LOWER(USE1_DESC) NOT LIKE '%church%'
  AND LOWER(USE1_DESC) NOT LIKE '%school%'
  AND LOWER(USE1_DESC) NOT LIKE '%forest%'
  AND LOWER(USE1_DESC) NOT LIKE '%state%'
  AND LOWER(USE1_DESC) NOT LIKE '%county%'
  AND LOWER(USE1_DESC) NOT LIKE '%util%'
  AND LOWER(USE1_DESC) NOT LIKE '%college%'
  AND LOWER(USE1_DESC) NOT LIKE '%cem%'
  AND LOWER(USE1_DESC) NOT LIKE '%common%'
  AND LOWER(USE1_DESC) NOT LIKE '%road%'
  AND LOWER(USE1_DESC) NOT LIKE '%fed%'
  AND LOWER(USE1_DESC) NOT LIKE '%tax%'
  AND LOWER(USE1_DESC) NOT LIKE '%dnr%'
  AND LOWER(USE1_DESC) NOT LIKE '%charit%'
  AND LOWER(USE1_DESC) NOT LIKE '%serv%'
  AND LOWER(USE1_DESC) NOT LIKE '%hosp%'
  AND LOWER(USE1_DESC) NOT LIKE '%park%'

  -- Agricultural
  AND LOWER(USE1_DESC) NOT LIKE '2__%'
  AND LOWER(USE1_DESC) NOT LIKE '%ag%'
  AND LOWER(USE1_DESC) NOT LIKE '%farm%'
  AND LOWER(USE1_DESC) NOT LIKE '%rural%'
)

, industrial AS (
SELECT
 USE1_DESC AS PropertyType
, centroid_long
, centroid_lat
FROM `datadive-142319.metrogis_parcels.2014_tax_parcel_data`
WHERE 

  -- Industrial
  (LOWER(USE1_DESC) LIKE '%ind%'
   OR LOWER(USE1_DESC) = '305 industrial')

  -- Residential
  AND LTRIM(LOWER(USE1_DESC)) NOT LIKE '1__%'
  AND LOWER(USE1_DESC) NOT LIKE '%residential%'
  AND LOWER(USE1_DESC) NOT LIKE '%res%'
  AND LOWER(USE1_DESC) NOT LIKE '%house%'
  AND LOWER(USE1_DESC) NOT LIKE '%condo%'
  AND LOWER(USE1_DESC) NOT LIKE '%apartment%'
  AND LOWER(USE1_DESC) NOT LIKE '%apt%'
  AND LOWER(USE1_DESC) NOT LIKE '%plex%'
  AND LOWER(USE1_DESC) NOT LIKE '%bungalo%'
  AND LOWER(USE1_DESC) NOT LIKE '%housing%'
  AND LOWER(USE1_DESC) NOT LIKE '%home%'
  AND LOWER(USE1_DESC) NOT LIKE '%family%'

  -- Public
  AND LOWER(USE1_DESC) NOT LIKE '9__%'
  AND LOWER(USE1_DESC) NOT LIKE '%public%'
  AND LOWER(USE1_DESC) NOT LIKE '%muni%'
  AND LOWER(USE1_DESC) NOT LIKE '%rail%'
  AND LOWER(USE1_DESC) NOT LIKE '%church%'
  AND LOWER(USE1_DESC) NOT LIKE '%school%'
  AND LOWER(USE1_DESC) NOT LIKE '%forest%'
  AND LOWER(USE1_DESC) NOT LIKE '%state%'
  AND LOWER(USE1_DESC) NOT LIKE '%county%'
  AND LOWER(USE1_DESC) NOT LIKE '%util%'
  AND LOWER(USE1_DESC) NOT LIKE '%college%'
  AND LOWER(USE1_DESC) NOT LIKE '%cem%'
  AND LOWER(USE1_DESC) NOT LIKE '%common%'
  AND LOWER(USE1_DESC) NOT LIKE '%road%'
  AND LOWER(USE1_DESC) NOT LIKE '%fed%'
  AND LOWER(USE1_DESC) NOT LIKE '%tax%'
  AND LOWER(USE1_DESC) NOT LIKE '%dnr%'
  AND LOWER(USE1_DESC) NOT LIKE '%charit%'
  AND LOWER(USE1_DESC) NOT LIKE '%serv%'
  AND LOWER(USE1_DESC) NOT LIKE '%hosp%'
  AND LOWER(USE1_DESC) NOT LIKE '%park%'

  -- Agricultural
  AND LOWER(USE1_DESC) NOT LIKE '2__%'
  AND LOWER(USE1_DESC) NOT LIKE '%ag%'
  AND LOWER(USE1_DESC) NOT LIKE '%farm%'
  AND LOWER(USE1_DESC) NOT LIKE '%rural%'

  -- Commercial
  AND LOWER(USE1_DESC) NOT LIKE '3__%'
  AND LOWER(USE1_DESC) NOT LIKE '%commercial%'
  AND LOWER(USE1_DESC) NOT LIKE '%machinery%'
  AND LOWER(USE1_DESC) NOT LIKE '%recreational%'
  AND LOWER(USE1_DESC) NOT LIKE '%golf%'
  AND LOWER(USE1_DESC) NOT LIKE '%coop%'
  

)

, public AS (
SELECT
 USE1_DESC AS PropertyType
, centroid_long
, centroid_lat
FROM `datadive-142319.metrogis_parcels.2014_tax_parcel_data`
WHERE 

   -- Public
   (LOWER(USE1_DESC) LIKE '9__%'
   OR LOWER(USE1_DESC) LIKE '%public%'
   OR LOWER(USE1_DESC) LIKE '%muni%'
   OR LOWER(USE1_DESC) LIKE '%rail%'
   OR LOWER(USE1_DESC) LIKE '%church%'
   OR LOWER(USE1_DESC) LIKE '%school%'
   OR LOWER(USE1_DESC) LIKE '%forest%'
   OR LOWER(USE1_DESC) LIKE '%state%'
   OR LOWER(USE1_DESC) LIKE '%county%'
   OR LOWER(USE1_DESC) LIKE '%util%'
   OR LOWER(USE1_DESC) LIKE '%college%'
   OR LOWER(USE1_DESC) LIKE '%cem%'
   OR LOWER(USE1_DESC) LIKE '%common%'
   OR LOWER(USE1_DESC) LIKE '%road%'
   OR LOWER(USE1_DESC) LIKE '%fed%'
   OR LOWER(USE1_DESC) LIKE '%tax%'
   OR LOWER(USE1_DESC) LIKE '%dnr%'
   OR LOWER(USE1_DESC) LIKE '%charit%'
   OR LOWER(USE1_DESC) LIKE '%serv%'
   OR LOWER(USE1_DESC) LIKE '%hosp%'
   OR LOWER(USE1_DESC) LIKE '%park%')

  -- Residential
  AND LTRIM(LOWER(USE1_DESC)) NOT LIKE '1__%'
  AND LOWER(USE1_DESC) NOT LIKE '%residential%'
  AND LOWER(USE1_DESC) NOT LIKE '%res%'
  AND LOWER(USE1_DESC) NOT LIKE '%house%'
  AND LOWER(USE1_DESC) NOT LIKE '%condo%'
  AND LOWER(USE1_DESC) NOT LIKE '%apartment%'
  AND LOWER(USE1_DESC) NOT LIKE '%apt%'
  AND LOWER(USE1_DESC) NOT LIKE '%plex%'
  AND LOWER(USE1_DESC) NOT LIKE '%bungalo%'
  AND LOWER(USE1_DESC) NOT LIKE '%housing%'
  AND LOWER(USE1_DESC) NOT LIKE '%home%'
  AND LOWER(USE1_DESC) NOT LIKE '%family%'

  -- Agricultural
  AND LOWER(USE1_DESC) NOT LIKE '2__%'
  AND LOWER(USE1_DESC) NOT LIKE '%ag%'
  AND LOWER(USE1_DESC) NOT LIKE '%farm%'
  AND LOWER(USE1_DESC) NOT LIKE '%rural%'

  -- Commercial
  AND LOWER(USE1_DESC) NOT LIKE '3__%'
  AND LOWER(USE1_DESC) NOT LIKE '%commercial%'
  AND LOWER(USE1_DESC) NOT LIKE '%machinery%'
  AND LOWER(USE1_DESC) NOT LIKE '%recreational%'
  AND LOWER(USE1_DESC) NOT LIKE '%golf%'
  AND LOWER(USE1_DESC) NOT LIKE '%coop%'

  -- Industrial
  AND LOWER(USE1_DESC) NOT LIKE '%ind%'
  AND LOWER(USE1_DESC) != '305 industrial'
)

SELECT
 ROW_NUMBER() OVER (ORDER BY lake.LakeName) AS ID
, lake.LakeName
, 2014 AS Year
, COUNT(residential.PropertyType) AS ResidentialCount
, COUNT(agriculture.PropertyType) AS AgriculturalCount
, COUNT(commercial.PropertyType) AS CommercialCount
, COUNT(industrial.PropertyType) AS IndustrialCount
, COUNT(public.PropertyType) AS PublicCount
FROM lake
  JOIN `datadive-142319.sds_xref.parcel_to_water` AS intersection ON lake.DNR_ID_SITE_NUMBER = intersection.MCES_Map_Code1
 
  LEFT JOIN residential ON intersection.parcel_centroid_long = residential.centroid_long
                       AND intersection.parcel_centroid_lat = residential.centroid_lat

  LEFT JOIN agriculture ON intersection.parcel_centroid_long = agriculture.centroid_long
                       AND intersection.parcel_centroid_lat = agriculture.centroid_lat

  LEFT JOIN commercial ON intersection.parcel_centroid_long = commercial.centroid_long
                       AND intersection.parcel_centroid_lat = commercial.centroid_lat

  LEFT JOIN industrial ON intersection.parcel_centroid_long = industrial.centroid_long
                       AND intersection.parcel_centroid_lat = industrial.centroid_lat

  LEFT JOIN public ON intersection.parcel_centroid_long = public.centroid_long
                       AND intersection.parcel_centroid_lat = public.centroid_lat
                 
GROUP BY lake.LakeName
ORDER BY lake.LakeName ASC

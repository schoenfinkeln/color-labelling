-- -- -- -- -- SQL QUERIES  -- -- -- -- --

-- Drop all views created
DROP VIEW IF EXISTS artigo_color_results;
DROP VIEW IF EXISTS algo_results;
DROP VIEW IF EXISTS image_color_tags;



/*
View defining color tags
*/

DROP VIEW IF EXISTS image_color_tags;
CREATE VIEW image_color_tags AS
SELECT 
tag.name AS tag_name, tagging.tag_id, COUNT(tagging.tag_id) AS tag_count, artresource.path, artresource.id, source.name AS folder
FROM 
tagging JOIN tag ON tagging.tag_id = tag.id 
	JOIN artresource ON tagging.resource_id = artresource.id
	JOIN source ON artresource.source_id = source.id 
WHERE tagging.tag_id IN
	(
	SELECT id 
	FROM tag 
	WHERE
		id < 1000000
		AND (
			   name ILIKE ANY(array['%blau%', '%rot%', '%gelb%', '%grün%', '%braun%', '%orange', '%beige%', 'ocker', '%grau%', 'schwarz', 'weiß', 'weiss', '%lila%', '%rosa%', '%pink%', '%violet%', 'dunkel%', 'hell%'])
		) AND (
			    name NOT ILIKE ALL(array['%brot%', '%boot%', 'blau%reiter', 'ROTUNDE', 'grotes%', 'eroti%', 'eroti%', 'grotte', '%rotterdam%', 'dunkelheit', 'helligkeit'])
			)
	) 
GROUP BY tagging.tag_id, artresource.path, artresource.id, tag.name, source.name
HAVING COUNT(tagging.tag_id) > 1
ORDER BY tag_count DESC;



/* 
View algo_results showing image_id and the ral_colors reference color id.
JOIN on ral.index and result.color_id
*/

DROP VIEW IF EXISTS algo_results;
CREATE VIEW algo_results AS 
SELECT DISTINCT color_tag_results.art_id AS image_id, ral_colors.reference AS color_id
FROM ral_colors JOIN color_tag_results ON ral_colors.index = color_tag_results.color_id



/* 
View artigo_color_results showing all art_id with color tags as ral_reference 
JOIN on tag_id is color_tag id
Depends on View image_color_tags 					
*/

DROP VIEW IF EXISTS artigo_color_results;
CREATE VIEW artigo_color_results AS
SELECT DISTINCT image_color_tags.id AS art_id, color_tags.ral_reference 
FROM image_color_tags JOIN color_tags ON image_color_tags.tag_id = color_tags.tag_id



/*
Select the ids, paths and reference_color_ids from images, that have been tagged with same reference color as algorithm predicts
*/

SELECT COUNT(*) FROM (
SELECT DISTINCT image_id, color_id, artresource.path
FROM algo_results, artigo_color_results, artresource
WHERE algo_results.image_id = artigo_color_results.art_id AND color_id = ral_reference AND artresource.id = art_id
) AS result_match;



/*
Count artigo color tagged images by color
*/

SELECT COUNT(art_id), ral_reference
FROM artigo_color_results 
GROUP BY ral_reference



/*
Count algorithm tagged images by color
*/

SELECT COUNT(image_id), color_id
FROM algo_results
GROUP BY color_id



/*
Average amount of color tags per image
*/

SELECT AVG(count) FROM (
SELECT COUNT(color_id), image_id
FROM algo_results
GROUP BY image_id
) AS average_colors



/*
Count algorithm predicting ARTigo per color
*/

SELECT COUNT(art_id), ral_reference
FROM algo_results, artigo_color_results, artresource
WHERE algo_results.image_id = artigo_color_results.art_id AND color_id = ral_reference AND artresource.id = art_id
GROUP BY ral_reference



/*
Results with German color name
*/

SELECT DISTINCT image_id, de_short, color_id
FROM algo_results JOIN ral_colors ON color_id = reference



/*
Image Dictionary
*/

DROP VIEW IF EXISTS image_dictionary;
CREATE VIEW image_dictionary AS
SELECT artresource.id AS art_id, title, artistname, date, path, source.url 
FROM artresource LEFT OUTER JOIN artresourcetitle ON artresource.id = artresourcetitle.resource_id
	LEFT OUTER JOIN artworkinfo ON artresource.id = artworkinfo.resource_id
	LEFT OUTER JOIN source ON artresource.source_id = source.id

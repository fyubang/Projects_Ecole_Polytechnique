create temp table id_keyword as
select papers.id, unnest(string_to_array(replace(cast(strip(
	to_tsvector('english', papers.name)) as text), '''', ''), ' ')) as keyword
from papers;

select year, keyword, count(*)
from id_keyword, papers, venue
where papers.id = id_keyword.id
and papers.venue = venue.id
group by year, keyword;
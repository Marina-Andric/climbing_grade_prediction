sql_mad_function = '''
    create or replace function mad_{1} (idt int, grade_id int, datum timestamp, grade_dev int) 
        returns numeric
        language plpgsql
    as 
    $$
    declare 
        res numeric;
    begin
    EXECUTE '	
        with route_data as (
            select * 
            from {2}
            where {0} = $1::integer and date < $2 and grade_id between $3 - $4 and $3 + $4
        ), q as (
            select avg(user_grade_id - grade_id) as mean_dif
            from route_data
        ), q1 as (
            select abs((user_grade_id-grade_id)) as mad
            -- select abs((user_grade_id-grade_id) - (select mean_dif from q)) as mad
            from route_data
        )	select round(avg(mad),2) from q1;' using idt, datum, grade_id, grade_dev
        into res;
        return res;
    end;
    $$;
'''

sql_mad_function_1year = '''
    create or replace function mad_{1}_1year (idt int, grade_id int, datum timestamp, grade_dev int, horizon varchar) 
        returns numeric
        language plpgsql
    as 
    $$
    declare 
        res numeric;
    begin
    EXECUTE E'	
        with route_data as (
            select * 
            from {2}
            where {0} = $1::integer and date < $2 and date >= $2 - interval \\'1.5 year\\' and grade_id between $3 - $4 and $3 + $4
        ), q as (
            select avg(user_grade_id - grade_id) as mean_dif
            from route_data
        ), q1 as (
            select abs((user_grade_id-grade_id)) as mad
            -- select abs((user_grade_id-grade_id) - (select mean_dif from q)) as mad
            from route_data
        )	select round(avg(mad),2) from q1;' using idt, datum, grade_id, grade_dev, horizon
        into res;
        return res;
    end;
    $$;
'''

sql_route_features = '''
    with q as (
        select t1.*
        from {4} as t1
	), route_data as (
		select * from q
		where route_id = {0} and date < '{1}'::timestamp
	),  route_data_1year as (
	    select * from q
		where route_id = {0} and date < '{1}'::timestamp and date >= '{1}'::timestamp - interval '1.5 year'
	),  route_data_1year_season as (
	    select * from route_data_1year
		where date_part('month', date) between date_part('month', '{1}'::timestamp) - 1 and date_part('month', '{1}'::timestamp) + 1
	), grade_diff_count as (
	    select (user_grade_id - grade_id) as grade_diff, count(*) as c_grade_diff
	    from route_data
	    group by user_grade_id - grade_id
     ), grade_diff_count_1year as (
	    select (user_grade_id - grade_id) as grade_diff, count(*) as c_grade_diff
	    from route_data_1year
	    group by user_grade_id - grade_id
     ), grade_diff_count_1year_season as (
	    select (user_grade_id - grade_id) as grade_diff, count(*) as c_grade_diff
	    from route_data_1year_season
	    group by user_grade_id - grade_id
     )
       select 
       {2} as grade_id,
       (case when (select count(*) from route_data) > 0 then 
            ((select count(*) from route_data)/((select count(*) from route_data) + {5})::numeric)*((select sum(grade_diff*c_grade_diff) from grade_diff_count)/(select count(*) from 
            route_data)::numeric)
        else 0 end) as route_dev_sign,

       (case when (select count(*) from route_data_1year) > 0 then 
            ((select count(*) from route_data_1year)/((select count(*) from route_data_1year) + {5})::numeric)*((select sum(grade_diff*c_grade_diff) from grade_diff_count_1year)/(select count(*) from 
            route_data_1year)::numeric)
        else 0 end) as route_dev_sign_1year,

       (case when (select count(*) from route_data_1year_season) > 0 then 
            ((select count(*) from route_data_1year_season)/((select count(*) from route_data_1year_season) + {5})::numeric)*((select sum(grade_diff*c_grade_diff) from grade_diff_count_1year_season)/(select count(*) from 
            route_data_1year_season)::numeric)
        else 0 end) as route_dev_sign_1year_season,        

        (select altitude from {4} where route_id = {0} limit 1) as altitude, 
        (select length from {4} where route_id = {0} limit 1) as length,
        (select rock_type from {4} where route_id = {0} limit 1) as rock_type
        '''

sql_user_features = '''
    with q as (
        select t1.*
        from {4} as t1
    ), 	user_data as (
        select * from q where user_id = {0} and date < '{1}'::timestamp
    ),  grade_diff_count as (
        select (user_grade_id - grade_id) as grade_diff, count(*) as c_grade_diff, grade_id
	    from user_data
	    group by user_grade_id - grade_id, grade_id
	),  user_data_season as (
	    select * 
	    from user_data
	    where date_part('month', date) between date_part('month', '{1}'::timestamp) - 1 and date_part('month', '{1}'::timestamp) + 1
	),  grade_diff_count_season as (
        select (user_grade_id - grade_id) as grade_diff, count(*) as c_grade_diff, grade_id
	    from user_data_season
	    group by user_grade_id - grade_id, grade_id	
    ),  user_freq_grade as (
        select user_id, grade_id, count(*) as n_grade_climbs 
        from user_data
        group by user_id, grade_id
        order by count(*) desc
        limit 1
    )  
       select 
        (case when (select count(*) from user_data where grade_id = {2}) > 0 then 
        ((select count(*) from user_data where grade_id = {2})/((select count(*) from user_data where grade_id = {2}) + {6})::numeric)*
        ((select sum(grade_diff*c_grade_diff) from grade_diff_count where grade_id = {2})/((select count(*) from user_data where grade_id = {2}))::numeric)
            when (select count(*) from user_data where grade_id between {2} - {5} and {2} + {5}) > 0 then 
        ((select count(*) from user_data where grade_id between {2} - {5} and {2} + {5})/((select count(*) from user_data where grade_id between {2} - {5} and {2} + {5}) + {6})::numeric)*
        ((select sum(grade_diff*c_grade_diff) from grade_diff_count where grade_id between {2} - {5} and {2} + {5})/((select count(*) from user_data where grade_id between {2} - {5} and {2} + {5}))::numeric)
        else 0.09 end) as user_dev_sign,
        
        (case when (select count(*) from user_data_season where grade_id = {2}) > 0 then 
        ((select count(*) from user_data_season where grade_id = {2})/((select count(*) from user_data_season where grade_id = {2}) + {6})::numeric)*
        ((select sum(grade_diff*c_grade_diff) from grade_diff_count_season where grade_id = {2})/((select count(*) from user_data_season where grade_id = {2}))::numeric)
            when (select count(*) from user_data_season where grade_id between {2} - {5} and {2} + {5}) > 0 then 
        ((select count(*) from user_data_season where grade_id between {2} - {5} and {2} + {5})/((select count(*) from user_data_season where grade_id between {2} - {5} and {2} + {5}) + {6})::numeric)*
        ((select sum(grade_diff*c_grade_diff) from grade_diff_count_season where grade_id between {2} - {5} and {2} + {5})/((select count(*) from user_data_season where grade_id between {2} - {5} and {2} + {5}))::numeric)
        else 0.09 end) as user_dev_sign_season,
        
        (case when date_part('month', '{1}'::timestamp) in (3, 4, 5) then 1
             when date_part('month', '{1}'::timestamp) in (6, 7, 8) then 2
              when date_part('month', '{1}'::timestamp) in (9, 10, 11) then 3
              when date_part('month', '{1}'::timestamp) in (12, 1, 2) then 4 end) as season,
        date_part('month', '{1}'::timestamp) as month,
        date_part('year', '{1}'::timestamp) as year           
        '''

sql_features_routes = '''
    drop table if exists {2};
    create table {2} as (
    WITH q 
         AS (
            select t1.*,
            (case when t2.test_route_id is not Null then 0 else 1 end) as "train"
             FROM  {0} t1
             left join {1} as t2 on t1.user_id = t2.user_id and t1.route_id = t2.test_route_id and t1.date = t2.test_route_date
             order by t1.date asc
    )
    select * from q);
     select * from {2};
'''



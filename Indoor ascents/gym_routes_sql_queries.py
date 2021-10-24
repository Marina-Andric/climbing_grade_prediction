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
    
            from route_data
        )	select round(avg(mad),2) from q1;' using idt, datum, grade_id, grade_dev
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
	), grade_diff_count as (
	    select (user_grade_id - grade_id) as grade_diff, count(*) as c_grade_diff
	    from route_data
	    group by user_grade_id - grade_id
     )
       select 
       {2} as grade_id,

       (case when (select count(*) from route_data) > 0 then 
            ((select count(*) from route_data)/((select count(*) from route_data) + {5})::numeric)*((select sum(grade_diff*c_grade_diff) from grade_diff_count)/(select count(*) from 
            route_data)::numeric)
        else 0 end) as route_dev_sign

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
        else 0 end) as user_dev_sign

        '''

sql_features_gym_routes = '''
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

sql_features_gym_routes_old = '''
    drop table if exists {2};
    create table {2} as (
    WITH q 
         AS (
            select t1.*,

            (case when t1.route_id = any (t2.train_route_id) then 1 
            when t1.route_id = any (t2.test_route_id) then 0 end) as "train"
             FROM  {0} t1
             left join {1} as t2 on t1.user_id = t2.user_id 

             order by t1.date asc
    ),	route_climb_count as (
        select train, route_id, count(*) as n_route_climbs
        from q
        group by train, route_id
    ),	user_climbs as (
        select train, user_id, count(*) as n_climbs
        from q
        group by train, user_id
    ), user_proposed as (
        select train, user_id, count(*) as n_proposed 
        from q
        where user_grade_id != grade_id
        group by user_id, train
    ),	user_stat as (
        select t1.train, t1.user_id, n_proposed, n_climbs, round(n_proposed/n_climbs::numeric, 2) as perc_proposed
        from user_climbs as t1
        join user_proposed as t2
        on t1.user_id = t2.user_id and t1.train = t2.train
        group by t1.train, t1.user_id, n_proposed, n_climbs
    ),	route_climbs as (
        select train, route_id, count(*) as n_climbs
        from q
        group by route_id, train
    ), 	route_proposed as (
        select train, route_id, count(*) as n_proposed
        from q
        where user_grade_id != grade_id
        group by route_id, train
    ),	route_stat as (
        select t1.train, t1.route_id, n_proposed, n_climbs, round(n_proposed/n_climbs::numeric, 2) as perc_proposed
        from route_climbs as t1
        join route_proposed as t2
        on t1.route_id = t2.route_id and t1.train = t2.train
        group by t1.train, t1.route_id, n_proposed, n_climbs
    
        ),	route_grading as (
            select train, route_id, user_grade_id - grade_id as grade_diff, count(*) as grade_diff_count
            from q
            group by train, route_id, user_grade_id - grade_id
        ),	route_grading_stats as (
            select t1.train, t1.route_id, round(sum((grade_diff * grade_diff_count) / n_route_climbs::numeric),
            2) as route_eval_bias
            from route_climb_count t1 
            join route_grading t2 on t1.route_id = t2.route_id and t1.train = t2.train
            group by t1.train, t1.route_id
        ),	users_per_grade_count as (
            select train, user_id, grade_id, count(*) as n_grade_climbs 
            from q
            group by user_id, grade_id, train
        ),	climbs_user_grading as (
            select train, user_id, user_grade_id, grade_id, user_grade_id - grade_id as grade_diff, count(*) as grade_diff_count
            from q
            group by train, user_id, user_grade_id, grade_id, user_grade_id - grade_id
        ),	user_grade_stats as (
            select t1.train, t1.user_id, t1.grade_id, round(sum((grade_diff * grade_diff_count) / n_grade_climbs::numeric),2) as user_grade_eval_bias
            from climbs_user_grading t1 
            join users_per_grade_count t2 on
            t1.user_id = t2.user_id and t1.grade_id = t2.grade_id and t1.train = t2.train
            group by t1.train, t1.user_id, t1.grade_id
        ), 	q_grades as (
            select train, 
             grade_id, 

             round(avg(user_grade_id)) as avg_user_grade_id_rp_per_grade
             from q
             group by train, grade_id --, user_red_point_id
        ),
         q3 
         AS (SELECT train, 
                    route_id, 
                    grade_id, 
                    Round(Avg(rating), 2)     AS avg_rating_per_route, 
                    Round(Avg(tries), 2)      AS avg_tries_per_route, 
                    Round(Avg(user_grade_id)) AS avg_user_grade_id_per_route, 
                    Round(Avg(user_climb_type), 2) AS avg_climb_type_per_route 
             FROM   q 
             GROUP  BY route_id, 
                       train, 
                       grade_id
             ORDER  BY route_id), 
         q4 
         AS (SELECT train, grade_id, 
                    Round(Avg(user_grade_id)) AS avg_user_grade_id_per_grade
             FROM   q 
             GROUP  BY grade_id, train 
             ORDER  BY grade_id), 
         q2 
         AS (SELECT t0.train, 
                    t0.date,
                    t0.route_id, 
                    t0.user_id, 
                    t0.grade_id, 
                    t0.route_setter_id,
                    t0.user_grade_id, 
                    q1.user_grade_eval_bias, 
                    q5.route_eval_bias,

			 		(case when q6.perc_proposed is Null then 0 else q6.perc_proposed end) as route_perc_proposed,
			 		(case when q7.perc_proposed is Null then  0 else q7.perc_proposed end) as user_perc_proposed,
			 		(case when q6.perc_proposed >= 0.4 then 1 else 0 end) as route_eligible,
			 		(case when q7.perc_proposed >= 0.4 then 1 else 0 end) as user_eligible,
                    avg_user_grade_id_per_grade, 
                    avg_rating_per_route, 
                    avg_tries_per_route, 
                    avg_user_grade_id_per_route, 
                    avg_climb_type_per_route, 
                    avg_user_grade_id_rp_per_grade
             FROM   q AS t0 
                    left join user_grade_stats q1
                           ON t0.user_id = q1.user_id 
                              AND t0.grade_id = q1.grade_id 
                                and t0.train = q1.train
                    join q3 
                      ON t0.route_id = q3.route_id 
                         AND t0.grade_id = q3.grade_id 
                        and t0.train = q3.train	
                    join q4 
                      ON t0.grade_id = q4.grade_id 
                        and t0.train = q4.train	
                    left join q_grades on t0.grade_id = q_grades.grade_id

                        and t0.train = q_grades.train
                    left join route_grading_stats as q5 on
                        t0.route_id = q5.route_id 
                        and t0.train = q5.train
			 		left join route_stat as q6 on 
			 			t0.route_id = q6.route_id and
			 			t0.train = q6.train
			 		left join user_stat as q7 on
			 			t0.user_id = q7.user_id and
			 			t0.train = q7.train
                    ORDER  BY t0.date asc )

    SELECT * 
    FROM q2 
     );
     select * from {2};
'''

sql_setter_features = '''
    with q as (
        select t1.*
        from {3} as t1
	), setter_data as (
		select * from q
		where route_setter_id = {0} and date < '{1}'::timestamp
	),  grade_diff_count as (
        select (user_grade_id - grade_id) as grade_diff, count(*) as c_grade_diff, grade_id
	    from setter_data

	    group by user_grade_id - grade_id, grade_id
    ) select 

    (case when (select count(*) from setter_data where grade_id = {2}) > 0 then 
        ((select count(*) from setter_data where grade_id = {2})/((select count(*) from setter_data where grade_id = {2}) + {5})::numeric)*
        ((select sum(grade_diff*c_grade_diff) from grade_diff_count where grade_id = {2})/((select count(*) from setter_data where grade_id = {2}))::numeric)
        when (select count(*) from setter_data where grade_id between {2} - {4} and {2} + {4}) > 0 then 
        ((select count(*) from setter_data where grade_id between {2} - {4} and {2} + {4})/((select count(*) from setter_data where grade_id between {2} - {4} and {2} + {4}) + {5})::numeric)*
        ((select sum(grade_diff*c_grade_diff) from grade_diff_count where grade_id between {2} - {4} and {2} + {4})/((select count(*) from setter_data where grade_id between {2} - {4} and {2} + 
        {4}))::numeric)
        else 0 end) as rs_dev_sign
'''
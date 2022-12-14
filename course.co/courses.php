<?php
// Initialize the session
session_start();
?>

<!DOCTYPE html>
<html lang="en">
    <head>
          <!-- courses css -->
   <link rel="stylesheet" href="css/course.css" />

    </head>
  

<?php include 'navbar.php'?>

  <br>
  <br>
  <br>
  <br>
  <br>

 <!--  Section Start -->
 <section class="features section-padding " id="features" >
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="section-title"><h2>Teach yourself <span>A new Skills</span></h2></div>
            </div>
        </div>
    </div>
    <div class="container">
        <div class="row justify-content-center" id="collection">
         
                <div class = " filter-button-group">
                    <button type = "button" class = "btn-1 m-2 text-dark active-filter-btn" data-filter = "*">All</button>
                    <button type = "button" class = "btn-1 m-2 text-dark" data-filter = ".best">Programming Language</button>
                    <button type = "button" class = "btn-1 m-2 text-dark" data-filter = ".feat">Graphics Design</button>
                    <button type = "button" class = "btn-1 m-2 text-dark" data-filter = ".new">Web Development</button>

                    <br>
                    <br>
                </div>
           
            <div class="container">
                <div class = "collection-list ">
                    <!-- single course -->
                    <div class="col-md-3  new">
                        <div class="course">
                            <a href="https://www.w3schools.com/html/default.asp" class="course-img" target="_blank">
                                <img src= "img/html.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" id="course-title" href="https://www.w3schools.com/html/default.asp" target="_blank" style="font-size:18px; text-align: center; color:#7857fe;">Complete course on HTML</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Web Development</span>
                                <span class="course-price course-free">Free</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course -->

                    <!-- single course -->
                    <div class="col-md-3  new">
                        <div class="course">
                            <a href="https://www.w3schools.com/css/default.asp" class="course-img" target="_blank">
                                <img src= "img/css.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="https://www.w3schools.com/css/default.asp" target="_blank" style="font-size:18px; text-align: center; color:#7857fe;">Introduction to CSS</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Web Development</span>
                                <span class="course-price course-free">Free</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course --> 
                     <!-- single course -->
                     <div class="col-md-3  new">
                        <div class="course">
                            <a href="https://www.w3schools.com/bootstrap4/default.asp" class="course-img" target="_blank">
                                <img src= "img/bootstrap.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="https://www.w3schools.com/bootstrap4/default.asp" target="_blank" style="font-size:18px; text-align: center;color:#7857fe;">Introduction to Bootstrap</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Web Development</span>
                                <span class="course-price course-free">Free</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course -->
                     
                    <!-- single course -->
                    <div class="col-md-3  new">
                        <div class="course">
                            <a href="https://www.youtube.com/watch?v=_a5j7KoflTs" class="course-img" target="_blank">
                                <img src= "img/sass.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="https://www.youtube.com/watch?v=_a5j7KoflTs" target="_blank" style="font-size:18px; text-align: center;color:#7857fe;">Introduction to Sass</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Web Development</span>
                                <span class="course-price course-free">Free</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course -->
                    <!-- single course -->
                    <div class="col-md-3 new">
                        <div class="course">
                            <a href="https://www.w3schools.com/js/default.asp" class="course-img" target="_blank">
                                <img src= "img/js.jpg" alt="JavaScriptLogo">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="https://www.w3schools.com/js/default.asp" target="_blank" style="font-size:18px; text-align: center;color:#7857fe;">JavaScript Tutorial</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Web Development</span>
                                <span class="course-price course-free">Free</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course -->

                    <!-- single course -->
                    <div class="col-md-3  new">
                        <div class="course">
                            <a href="https://www.udemy.com/course/professional-rails-5-development-course/" class="course-img" target="_blank">
                                <img src= "img/rails.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="https://www.udemy.com/course/professional-rails-5-development-course/" target="_blank" style="font-size:18px; text-align: center;color:#7857fe;">Ruby on Rails5</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Web Development</span>
                                <span class="course-price course-premium">Premium</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course --> 
                     <!-- single course -->
                     <div class="col-md-3  new">
                        <div class="course">
                            <a href="https://www.w3schools.com/jquery/default.asp" class="course-img" target="_blank">
                                <img src= "img/jquery.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="https://www.w3schools.com/jquery/default.asp" target="_blank" style="font-size:18px; text-align: center;color:#7857fe;"> JQuery Tutorial </a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Web Development</span>
                                <span class="course-price course-free">Free</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course -->
                     
                    <!-- single course -->
                    <div class="col-md-3  new">
                        <div class="course">
                            <a href=" https://reactjs.org/community/courses.html" class="course-img" target="_blank">
                                <img src= "img/react.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href=" https://reactjs.org/community/courses.html" target="_blank" style="font-size:18px; text-align: center; color:#7857fe;">React Complete Course</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Web Development</span>
                                <span class="course-price course-premium">Premium</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course -->


    <!-- single course -->
    <div class="col-md-3  best">
                        <div class="course">
                            <a href="https://www.javacodegeeks.com/2021/10/the-premium-java-programming-certification-bundle.html" class="course-img" target="_blank">
                                <img src= "img/java.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="https://www.javacodegeeks.com/2021/10/the-premium-java-programming-certification-bundle.html" target="_blank" style="font-size:18px; text-align: center;color:#7857fe;">Core Java</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Web Development</span>
                                <span class="course-price course-premium">Premium</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course -->

                    <!-- single course -->
                    <div class="col-md-3  best">
                        <div class="course">
                            <a href="https://www.codecademy.com/learn/paths/computer-science?g_network=g&g_device=c&g_adid=526540401325&g_keyword=python%20programming%20for%20beginners&g_acctid=243-039-7011&g_adtype=search&g_adgroupid=102650142433&g_keywordid=kwd-301260171534&g_campaign=ROW+Career+Path%3A+Pro+-+Exact&g_campaignid=10074200774&utm_id=t_kwd-301260171534:ag_102650142433:cp_10074200774:n_g:d_c&utm_term=python%20programming%20for%20beginners&utm_campaign=ROW%20Career%20Path%3A%20Pro%20-%20Exact&utm_source=google&u" class="course-img" target="_blank">
                                <img src= "img/python.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="https://www.codecademy.com/learn/paths/computer-science?g_network=g&g_device=c&g_adid=526540401325&g_keyword=python%20programming%20for%20beginners&g_acctid=243-039-7011&g_adtype=search&g_adgroupid=102650142433&g_keywordid=kwd-301260171534&g_campaign=ROW+Career+Path%3A+Pro+-+Exact&g_campaignid=10074200774&utm_id=t_kwd-301260171534:ag_102650142433:cp_10074200774:n_g:d_c&utm_term=python%20programming%20for%20beginners&utm_campaign=ROW%20Career%20Path%3A%20Pro%20-%20Exact&utm_source=google&u" target="_blank" style="font-size:18px; text-align: center;color:#7857fe;">Python Programming</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Programming</span>
                                <span class="course-price course-free">Free</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course --> 
                     <!-- single course -->
                     <div class="col-md-3  best">
                        <div class="course">
                            <a href="https://www.w3schools.com/cs/index.php" class="course-img" target="_blank">
                                <img src= "img/c.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="https://www.w3schools.com/cs/index.php" target="_blank" style="font-size:18px; text-align: center; color:#7857fe;">C# Tutorial</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Programming</span>
                                <span class="course-price course-Free">Free</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course -->
                     
                    <!-- single course -->
                    <div class="col-md-3  best">
                        <div class="course">
                            <a href="https://www.w3schools.com/cpp/default.asp" class="course-img" target="_blank">
                                <img src= "img/Cplus.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="https://www.w3schools.com/cpp/default.asp" target="_blank" style="font-size:18px; text-align: center; color:#7857fe;">C++ programming</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Programming</span>
                                <span class="course-price course-free">Free</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course -->
                    <!-- single course -->
                    <div class="col-md-3  best">
                        <div class="course">
                            <a href="https://www.w3schools.com/php/default.asp" class="course-img" target="_blank">
                                <img src= "img/php.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="https://www.w3schools.com/php/default.asp" target="_blank" style="font-size:18px; text-align: center; color:#7857fe;">PHP  Tutorial</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Server Side</span>
                                <span class="course-price course-free">Free</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course -->

                    <!-- single course -->
                    <div class="col-md-3  best">
                        <div class="course">
                            <a href="https://www.w3schools.com/sql/" class="course-img" target="_blank">
                                <img src= "img/sql.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="https://www.w3schools.com/sql/" target="_blank" style="font-size:18px; text-align: center; color:#7857fe;">SQL Tutorial</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Server Side</span>
                                <span class="course-price course-free">Free</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course --> 
                     <!-- single course -->
                     <div class="col-md-3  best">
                        <div class="course">
                            <a href="https://medium.com/@ipenywis/learn-node-js-from-scratch-in-1-hour-crash-course-c83479d7fb46" class="course-img" target="_blank">
                                <img src= "img/nodejs.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="https://medium.com/@ipenywis/learn-node-js-from-scratch-in-1-hour-crash-course-c83479d7fb46" target="_blank" style="font-size:18px; text-align: center; color:#7857fe;">Node.Js Crash Course</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Server Side</span>
                                <span class="course-price course-free">Free</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course -->
                     
                    <!-- single course -->
                    <div class="col-md-3  best">
                        <div class="course">
                            <a href="https://www.w3schools.com/git/default.asp" class="course-img" target="_blank">
                                <img src= "img/git.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="https://www.w3schools.com/git/default.asp" target="_blank" style="font-size:18px; text-align: center; color:#7857fe;">Mastering Git</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Server Side</span>
                                <span class="course-price course-free">Free</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course -->    
                     <!-- single course -->
                     <div class="col-md-3 feat">
                        <div class="course">
                            <a href="https://bit.ly/3nckQOi" class="course-img" target="_blank">
                                <img src= "img/maya.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="https://bit.ly/3nckQOi" target="_blank" style="font-size:18px; text-align: center; color:#7857fe;">Autodesk Maya Tutorial</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Animation and Motion Graphics</span>
                                <span class="course-price course-free">Free</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course -->
                     
                    <!-- single course -->
                    <div class="col-md-3  feat">
                        <div class="course">
                            <a href="https://www.photoshopessentials.com/" class="course-img" target="_blank">
                                <img src= "img/ps.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="https://www.photoshopessentials.com/" target="_blank" style="font-size:18px; text-align: center; color:#7857fe;">Adobe Photoshop</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Animation and Motion Graphics</span>
                                <span class="course-price course-premium">Premium</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course -->  
                     <!-- single course -->
                     <div class="col-md-3  feat">
                        <div class="course">
                            <a href="https://www.pluralsight.com/courses/blender-fundamentals?aid=7010a000002LUv7AAG&promo=&utm_source=non_branded&utm_medium=digital_paid_search_google&utm_campaign=XYZ_APAC_Dynamic&utm_content=&cq_cmp=1576650374&gclid=CjwKCAiAgbiQBhAHEiwAuQ6BkvWvom-FyZFKZBlYwE2tAhqSc5f4B9VljD87q1VjggSneQIpajwHqxoCta0QAvD_BwE" class="course-img" target="_blank">
                                <img src= "img/blender.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="https://www.pluralsight.com/courses/blender-fundamentals?aid=7010a000002LUv7AAG&promo=&utm_source=non_branded&utm_medium=digital_paid_search_google&utm_campaign=XYZ_APAC_Dynamic&utm_content=&cq_cmp=1576650374&gclid=CjwKCAiAgbiQBhAHEiwAuQ6BkvWvom-FyZFKZBlYwE2tAhqSc5f4B9VljD87q1VjggSneQIpajwHqxoCta0QAvD_BwE" target="_blank" style="font-size:18px; text-align: center; color:#7857fe;">Blender course</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Animation and Motion Graphics</span>
                                <span class="course-price course-free">Free</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course -->
                     
                    <!-- single course -->
                    <div class="col-md-3  feat">
                        <div class="course">
                            <a href="https://www.courseduck.com/programming/cinema-4d/" class="course-img" target="_blank">
                                <img src= "img/cinema.jpg" alt="">
                                <i class="course-link-icon fa fa-link"></i>
                            </a>
                             <a class="course-title" href="hhttps://www.courseduck.com/programming/cinema-4d/" target="_blank" style="font-size:18px; text-align: center; color:#7857fe;">Cinema 4D Courses</a>
                            <div class="course-details" style="margin: 5px;">
                                <span class="course-category">Animation and Motion Graphics</span>
                                <span class="course-price course-premium">Premium</span>
                            </div>
                        </div>
                    </div>
                     <!-- single course -->               
                    


                </div>
            </div>
            

        </div>
    </div>
<br>
<br>
    
  </section>


 <!-- ########################################################################################## -->
 <!-- ########################################################################################## -->
 <!-- ########################################################################################## -->


<!-- footer -->
 <?php include 'footer.php'?>
	
    <!-- jquery -->
    <script src = "js/jquery-3.6.0.js"></script>
    <!-- isotope js -->
    <script src="https://unpkg.com/isotope-layout@3/dist/isotope.pkgd.js"></script>
    <!-- bootstrap js -->
    <!-- <script src = "bootstrap-5.0.2-dist/js/bootstrap.min.js"></script> -->
    <!-- custom js -->
    <script src = "js/script.js"></script>

	</body>
</html>
